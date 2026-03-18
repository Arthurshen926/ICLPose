"""
DINO+SD融合特征提取器
封装从GeoAware-SC提取的特征提取流程
"""
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path

# 设置缓存路径
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))


class FusedFeatureExtractor:
    """
    DINO + Stable Diffusion 融合特征提取器
    
    使用方法:
        extractor = FusedFeatureExtractor(device='cuda')
        features = extractor.extract(image_path)  # (C, H, W) torch.Tensor
    """
    
    def __init__(self, 
                 device='cuda',
                 sd_image_size=480,
                 aggregator_weights=None):
        """
        初始化特征提取器
        
        Args:
            device: 计算设备
            sd_image_size: SD特征提取的短边尺寸
            aggregator_weights: AggregationNetwork权重路径
        """
        self.device = device
        self.sd_image_size = sd_image_size
        
        # 加载模型
        self._load_models(aggregator_weights)
        
        # 禁用梯度
        torch.set_grad_enabled(False)
    
    def _load_models(self, aggregator_weights=None):
        """加载所有需要的模型"""
        print("[FusedFeatureExtractor] 加载模型...")
        
        # 加载SD模型
        from .extractor_sd import load_model
        self.sd_model, self.sd_aug = load_model(
            diffusion_ver='v1-5', 
            image_size=self.sd_image_size, 
            num_timesteps=50, 
            block_indices=[2, 5, 8, 11]
        )
        print("  ✓ Stable Diffusion模型加载完成")
        
        # 加载DINO模型
        from .extractor_dino import ViTExtractor
        self.extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device=self.device)
        print("  ✓ DINOv2模型加载完成")
        
        # 加载聚合网络
        from .projection_network import AggregationNetwork
        self.aggre_net = AggregationNetwork(
            feature_dims=[640, 1280, 1280, 768], 
            projection_dim=768, 
            device=self.device
        )
        
        # 加载预训练权重
        if aggregator_weights is None:
            # 默认权重路径
            default_weights = Path(__file__).parent.parent / 'reference/GeoAware-SC/results_spair/best_856.PTH'
            if default_weights.exists():
                aggregator_weights = str(default_weights)
        
        if aggregator_weights and Path(aggregator_weights).exists():
            self.aggre_net.load_pretrained_weights(torch.load(aggregator_weights))
            print(f"  ✓ AggregationNetwork加载权重: {aggregator_weights}")
        else:
            print("  ⚠ AggregationNetwork使用随机权重")
        
        print("[FusedFeatureExtractor] 模型加载完成")
    
    def extract(self, image_input, return_numpy=False):
        """
        提取融合特征
        
        Args:
            image_input: 图像路径(str/Path)或PIL.Image
            return_numpy: 是否返回numpy数组
            
        Returns:
            fused_feat: (C, H, W) 融合特征, C=768
        """
        # 加载图像
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input
        
        # 计算SD输入尺寸（保宽高比，短边对齐 sd_image_size）
        orig_w, orig_h = img.size
        scale = self.sd_image_size / min(orig_w, orig_h)
        sd_w = int(round(orig_w * scale))
        sd_h = int(round(orig_h * scale))

        # ── SD 特征提取 ──────────────────────────────────────────────────────
        # 重要: ODISE 的 aug 内含 ResizeShortestEdge(short_edge_length=sd_image_size)。
        # 若我们先做反射填充再送入, ODISE 会对填充后的图像重新 resize,
        # 导致宽度变为非 64 整除值, SD UNet 的内部 zero-pad 产生右侧伪影。
        #
        # 正确策略: 不做外部填充, 直接按 sd_w×sd_h 送入 ODISE。
        # ODISE aug 不会改变其尺寸 (短边已等于 sd_image_size)。
        # SD UNet 内部会在右/下 zero-pad 到 64 整除, 我们从 feature shape
        # 动态推算 zero-pad 范围并裁掉即可。
        img_sd_input = img.resize((sd_w, sd_h), Image.Resampling.LANCZOS)

        from .extractor_sd import process_features_and_mask
        feats_sd = process_features_and_mask(
            self.sd_model, self.sd_aug, img_sd_input,
            mask=False, raw=True
        )
        del feats_sd['s2']  # 不使用s2

        # 裁掉 SD UNet 内部 zero-pad 对应的 feature 行/列
        sd_padded_h = int(math.ceil(sd_h / 64) * 64)
        sd_padded_w = int(math.ceil(sd_w / 64) * 64)
        for key in ['s3', 's4', 's5']:
            if key not in feats_sd:
                continue
            _, _, fh, fw = feats_sd[key].shape
            factor_h = sd_padded_h // fh
            factor_w = sd_padded_w // fw
            valid_fh = sd_h // factor_h
            valid_fw = sd_w // factor_w
            if valid_fh < fh or valid_fw < fw:
                feats_sd[key] = feats_sd[key][:, :, :valid_fh, :valid_fw]

        # ── DINO 特征提取（尺寸对齐到 14 的倍数）────────────────────────────
        dino_w = int(math.ceil(sd_w / 14) * 14)
        dino_h = int(math.ceil(sd_h / 14) * 14)
        img_dino_input = img.resize((dino_w, dino_h), Image.Resampling.BILINEAR)
        img_batch = self.extractor_vit.preprocess_pil(img_dino_input)

        tokens_h, tokens_w = dino_h // 14, dino_w // 14
        feats_dino = self.extractor_vit.extract_descriptors(
            img_batch.to(self.device), layer=11, facet='token'
        )
        feats_dino = feats_dino.permute(0, 1, 3, 2).reshape(1, -1, tokens_h, tokens_w)

        # 对齐空间尺寸并融合
        desc_gathered = torch.cat([
            F.interpolate(feats_sd['s3'], size=(tokens_h, tokens_w), mode='bilinear', align_corners=False),
            F.interpolate(feats_sd['s4'], size=(tokens_h, tokens_w), mode='bilinear', align_corners=False),
            F.interpolate(feats_sd['s5'], size=(tokens_h, tokens_w), mode='bilinear', align_corners=False),
            feats_dino
        ], dim=1)
        
        # 聚合
        fused = self.aggre_net(desc_gathered)
        fused = fused / (torch.linalg.norm(fused, dim=1, keepdim=True) + 1e-8)
        
        # 移除batch维度
        fused = fused.squeeze(0)  # (C, H, W)
        
        if return_numpy:
            return fused.cpu().numpy()
        return fused.cpu()
    
    def extract_batch(self, image_paths, batch_size=4):
        """
        批量提取特征
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批量大小 (当前实现逐个处理)
            
        Returns:
            features: list of (C, H, W) tensors
        """
        features = []
        for path in image_paths:
            feat = self.extract(path)
            features.append(feat)
        return features


def create_extractor(device='cuda', aggregator_weights=None):
    """
    创建特征提取器的便捷函数
    
    Args:
        device: 计算设备
        aggregator_weights: 聚合网络权重路径
        
    Returns:
        FusedFeatureExtractor 实例
    """
    return FusedFeatureExtractor(
        device=device,
        aggregator_weights=aggregator_weights
    )
