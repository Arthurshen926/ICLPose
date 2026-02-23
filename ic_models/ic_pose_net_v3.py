"""
ICPoseNetV3: Iterative Render-and-Compare Pose Estimation
==========================================================
核心架构: 多尺度残差 → Scale-Gated Fusion → ConvGRU → Dual-Head (Pose + Flow)

数据流:
  1. NetVLAD Top-1 → 初始位姿 P₀
  2. for k = 1..K:
       a. 用 P_{k-1} 渲染多尺度 3DGS 特征图 + 深度图
       b. 计算 query vs rendered 的多尺度残差
       c. Scale-Gated Fusion → 统一残差 E_fused
       d. ConvGRU(E_fused, h_{k-1}) → h_k
       e. Pose Head: h_k → Δξ ∈ se(3)
       f. Flow Head: h_k → (Δu, Δv, confidence)
       g. P_k = exp(Δξ) · P_{k-1}
  3. 输出最终位姿 P_K

参考:
  - RAFT (Teed & Deng, ECCV 2020)
  - DROID-SLAM (Teed & Deng, NeurIPS 2021)
  - iNeRF (Yen-Chen et al., IROS 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from modules.lie_algebra import se3_exp, pose_inverse
from modules.conv_gru import ConvGRUBlock
from modules.dynamic_feature_selector import DynamicFeatureSelector
from modules.dual_head import DualHead


class ICPoseNetV3(nn.Module):
    """
    迭代 Render-and-Compare 位姿估计网络 (V3)
    
    核心组件:
      - DynamicFeatureSelector: 多尺度残差 → Scale-Gated Fusion
      - ConvGRUBlock: 循环更新隐藏状态
      - DualHead: Pose Head (隐式) + Flow Head (显式)
      - se(3) 指数映射: 位姿增量更新
    
    输入:
      - query_feats: {scale_name: (B, D_i, H_i, W_i)} 查询图像多尺度特征
      - initial_pose: (B, 4, 4) 初始位姿 (w2c)
      - renderer: MultiScaleRenderer 实例 (不是本模块的参数)
    
    输出:
      - 每步的位姿、flow、gate值等
    
    Args:
        scale_configs: 各尺度配置 [{'name': ..., 'feat_dim': ...}, ...]
        hidden_dim: ConvGRU 隐藏状态维度
        output_resolution: 统一的 spatial 分辨率
        num_iters: 训练时的默认迭代次数
        residual_mode: 残差计算方式 ('concat', 'subtract', 'correlation')
        residual_out_dim: 残差输出维度
    """
    
    def __init__(
        self,
        scale_configs: List[Dict] = None,
        hidden_dim: int = 128,
        output_resolution: Tuple[int, int] = (35, 46),
        num_iters: int = 4,
        residual_mode: str = 'concat',
        residual_out_dim: int = 64,
        pose_mlp_dim: int = 256,
        flow_intermediate_dim: int = 64,
        # FlowToPose 参数 (V8)
        flow_to_pose_intrinsics: Dict[str, float] = None,
        flow_to_pose_damping: float = 1e-3,
    ):
        super().__init__()
        
        # 默认 4 尺度配置 (压缩后维度)
        if scale_configs is None:
            scale_configs = [
                {'name': 'coarse',    'feat_dim': 32,  'resolution': (7, 10)},
                {'name': 'mid',       'feat_dim': 64,  'resolution': (15, 20)},
                {'name': 'fine_sd',   'feat_dim': 64,  'resolution': (35, 46)},
                {'name': 'fine_dino', 'feat_dim': 64,  'resolution': (35, 46)},
            ]
        
        self.scale_configs = scale_configs
        self.hidden_dim = hidden_dim
        self.output_resolution = output_resolution
        self.num_iters = num_iters
        
        # ① Dynamic Feature Selector
        self.feature_selector = DynamicFeatureSelector(
            scale_configs=scale_configs,
            output_resolution=output_resolution,
            hidden_dim=hidden_dim,
            residual_mode=residual_mode,
            residual_out_dim=residual_out_dim,
        )
        
        # ② ConvGRU Update Block
        self.update_block = ConvGRUBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )
        
        # ③ Dual Head (Flow + FlowToPose geometric)
        self.dual_head = DualHead(
            hidden_dim=hidden_dim,
            pose_mlp_dim=pose_mlp_dim,
            flow_intermediate_dim=flow_intermediate_dim,
            flow_to_pose_intrinsics=flow_to_pose_intrinsics,
            flow_to_pose_damping=flow_to_pose_damping,
        )
        
        # 可学习的隐藏状态初始化 (比 zeros 更好)
        self.hidden_init = nn.Parameter(
            torch.zeros(1, hidden_dim, output_resolution[0], output_resolution[1])
        )
        nn.init.normal_(self.hidden_init, std=0.01)
        
        # 统计
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ICPoseNetV3] Initialized:")
        print(f"  Scales: {[c['name'] for c in scale_configs]}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output resolution: {output_resolution}")
        print(f"  Default iters: {num_iters}")
        print(f"  Residual mode: {residual_mode}")
        print(f"  FlowToPose: {'enabled' if flow_to_pose_intrinsics else 'disabled'}")
        print(f"  Trainable params: {n_params:,}")
    
    def forward(
        self,
        query_feats: Dict[str, torch.Tensor],
        initial_pose: torch.Tensor,
        renderer=None,
        intrinsics: Dict[str, float] = None,
        depth_gt: torch.Tensor = None,
        num_iters: int = None,
        return_all_steps: bool = True,
        detach_between_iters: bool = True,
    ) -> Dict[str, object]:
        """
        前向传播: 迭代 Render-and-Compare
        
        Args:
            query_feats: {scale_name: (B, D_i, H_i, W_i)} 预提取的查询特征
            initial_pose: (B, 4, 4) 初始位姿 (w2c)
            renderer: MultiScaleRenderer 实例 (推理时渲染)
            intrinsics: {'fx':, 'fy':, 'cx':, 'cy':} (用于 flow GT 计算)
            depth_gt: (B, H, W) GT 深度图 (用于 FlowToPose 几何转换)
            num_iters: 迭代次数 (None 使用默认值)
            return_all_steps: 是否返回所有步的中间结果
            detach_between_iters: 是否在迭代间 detach 位姿
            
        Returns:
            dict:
                'poses': [(B, 4, 4)] 每步的位姿 (K+1 个: P₀ 到 P_K)
                'xi_list': [(B, 6)] 每步的 se(3) 增量
                'flow_list': [(B, 2, H, W)] 每步的 flow 预测
                'log_conf_list': [(B, 1, H, W)] 每步的置信度
                'gate_values_list': [dict] 每步的 gate 值
                'final_pose': (B, 4, 4) 最终位姿
                'hidden': (B, C_h, H, W) 最后的隐藏状态
        """
        if num_iters is None:
            num_iters = self.num_iters
        
        B = initial_pose.shape[0]
        device = initial_pose.device
        
        # L2 归一化查询特征 (匹配渲染器的 norm_feat_after_render)
        query_feats_norm = {}
        for k_name, feat in query_feats.items():
            query_feats_norm[k_name] = F.normalize(feat, p=2, dim=1)
        
        # 初始化隐藏状态
        hidden = self.hidden_init.expand(B, -1, -1, -1).contiguous()
        
        # 初始位姿
        current_pose = initial_pose.clone()
        
        # 收集每步结果
        poses = [current_pose.clone()]
        xi_list = []
        flow_list = []
        log_conf_list = []
        gate_values_list = []
        
        scale_names = [c['name'] for c in self.scale_configs]
        
        for k in range(num_iters):
            # ① 渲染多尺度特征图 (from current_pose)
            if renderer is not None:
                rendered_feats = self._render_features(
                    renderer, current_pose, scale_names
                )
            else:
                raise ValueError(
                    "renderer must be provided for forward pass. "
                    "Use forward_with_prerendered() for pre-rendered features."
                )
            
            # ② Dynamic Feature Selector: 计算残差 + 门控融合
            fused_residual, gate_values = self.feature_selector(
                query_feats_norm, rendered_feats
            )
            # fused_residual: (B, hidden_dim, H_out, W_out)
            
            # ③ ConvGRU 更新隐藏状态
            hidden = self.update_block(fused_residual, hidden)
            
            # ④ Dual Head: 解码 flow + 几何推导位姿
            head_output = self.dual_head(hidden, depth=depth_gt)
            xi = head_output['xi']              # (B, 6)
            flow = head_output['flow']          # (B, 2, H, W)
            log_conf = head_output['log_confidence']  # (B, 1, H, W)
            
            # ⑤ SE(3) 位姿更新: P_k = exp(Δξ) · P_{k-1}
            delta_T = se3_exp(xi)               # (B, 4, 4)
            current_pose = delta_T @ current_pose
            
            # Detach pose between iterations for cleaner gradients
            # (each iteration independently predicts correction from current residual)
            if detach_between_iters and k < num_iters - 1:
                current_pose = current_pose.detach()
            
            # 收集结果
            poses.append(current_pose.clone())
            xi_list.append(xi)
            flow_list.append(flow)
            log_conf_list.append(log_conf)
            gate_values_list.append(gate_values)
        
        return {
            'poses': poses,
            'xi_list': xi_list,
            'flow_list': flow_list,
            'log_conf_list': log_conf_list,
            'gate_values_list': gate_values_list,
            'final_pose': current_pose,
            'hidden': hidden,
        }
    
    def forward_with_prerendered(
        self,
        query_feats: Dict[str, torch.Tensor],
        rendered_feats_list: List[Dict[str, torch.Tensor]],
        initial_pose: torch.Tensor,
        num_iters: int = None,
    ) -> Dict[str, object]:
        """
        使用预渲染的特征进行前向传播 (训练时避免在 backward 中重复渲染)
        
        在训练中的典型用法:
          1. 预先为每步渲染特征 (在 no_grad 块中)
          2. 调用此函数进行前向 + 反传
        
        Args:
            query_feats: {scale_name: (B, D_i, H_i, W_i)}
            rendered_feats_list: length=num_iters, 
                每个元素是 {scale_name: (B, D_i, H_i, W_i)}
            initial_pose: (B, 4, 4)
        """
        if num_iters is None:
            num_iters = min(self.num_iters, len(rendered_feats_list))
        
        B = initial_pose.shape[0]
        
        # L2 归一化查询特征 (匹配渲染器的 norm_feat_after_render)
        query_feats_norm = {}
        for k_name, feat in query_feats.items():
            query_feats_norm[k_name] = F.normalize(feat, p=2, dim=1)
        
        hidden = self.hidden_init.expand(B, -1, -1, -1).contiguous()
        current_pose = initial_pose.clone()
        
        poses = [current_pose.clone()]
        xi_list = []
        flow_list = []
        log_conf_list = []
        gate_values_list = []
        
        for k in range(num_iters):
            rendered_feats = rendered_feats_list[k]
            
            fused_residual, gate_values = self.feature_selector(
                query_feats_norm, rendered_feats
            )
            
            hidden = self.update_block(fused_residual, hidden)
            head_output = self.dual_head(hidden)
            
            xi = head_output['xi']
            flow = head_output['flow']
            log_conf = head_output['log_confidence']
            
            delta_T = se3_exp(xi)
            current_pose = delta_T @ current_pose
            
            poses.append(current_pose.clone())
            xi_list.append(xi)
            flow_list.append(flow)
            log_conf_list.append(log_conf)
            gate_values_list.append(gate_values)
        
        return {
            'poses': poses,
            'xi_list': xi_list,
            'flow_list': flow_list,
            'log_conf_list': log_conf_list,
            'gate_values_list': gate_values_list,
            'final_pose': current_pose,
            'hidden': hidden,
        }
    
    @torch.no_grad()
    def _render_features(
        self,
        renderer,
        poses_w2c: torch.Tensor,
        scale_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        调用 MultiScaleRenderer 渲染所有尺度的特征图
        
        Args:
            renderer: MultiScaleRenderer
            poses_w2c: (B, 4, 4) batch of w2c poses
            scale_names: 需要渲染的尺度列表
        
        Returns:
            {scale_name: (B, D_i, H_i, W_i)}
        """
        B = poses_w2c.shape[0]
        
        result = {}
        for name in scale_names:
            feat_list = []
            for b in range(B):
                r = renderer.render_scale(name, poses_w2c[b])
                feat_list.append(r['feature_map'])  # (D, H, W)
            result[name] = torch.stack(feat_list, dim=0)  # (B, D, H, W)
        
        return result


class ICPoseNetV3Config:
    """
    ICPoseNetV3 的默认配置

    使用原始(未压缩)特征维度时的配置
    """
    
    @staticmethod
    def raw_features() -> dict:
        """使用原始特征维度 (无压缩)"""
        return {
            'scale_configs': [
                {'name': 'coarse',    'feat_dim': 1280, 'resolution': (7, 10)},
                {'name': 'mid',       'feat_dim': 1280, 'resolution': (15, 20)},
                {'name': 'fine_sd',   'feat_dim': 640,  'resolution': (35, 46)},
                {'name': 'fine_dino', 'feat_dim': 768,  'resolution': (35, 46)},
            ],
            'hidden_dim': 128,
            'output_resolution': (35, 46),
            'num_iters': 4,
            'residual_mode': 'concat',
            'residual_out_dim': 64,
        }
    
    @staticmethod
    def compressed_features() -> dict:
        """使用压缩后的特征维度"""
        return {
            'scale_configs': [
                {'name': 'coarse',    'feat_dim': 32,  'resolution': (7, 10)},
                {'name': 'mid',       'feat_dim': 64,  'resolution': (15, 20)},
                {'name': 'fine_sd',   'feat_dim': 64,  'resolution': (35, 46)},
                {'name': 'fine_dino', 'feat_dim': 64,  'resolution': (35, 46)},
            ],
            'hidden_dim': 128,
            'output_resolution': (35, 46),
            'num_iters': 6,
            'residual_mode': 'concat',
            'residual_out_dim': 64,
        }
    
    @staticmethod
    def lightweight() -> dict:
        """轻量版 (用于快速验证)"""
        return {
            'scale_configs': [
                {'name': 'coarse',    'feat_dim': 1280, 'resolution': (7, 10)},
                {'name': 'fine_dino', 'feat_dim': 768,  'resolution': (35, 46)},
            ],
            'hidden_dim': 64,
            'output_resolution': (35, 46),
            'num_iters': 3,
            'residual_mode': 'subtract',
            'residual_out_dim': 64,
        }
