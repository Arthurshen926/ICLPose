"""
特征压缩器
封装AutoEncoder进行特征降维
"""
import torch
import numpy as np
from pathlib import Path
from .autoencoder import AutoencoderFlexible, FEATURE_CONFIGS


class FeatureCompressor:
    """
    特征压缩器
    
    使用预训练的AutoEncoder将768维融合特征压缩为256维
    
    使用方法:
        compressor = FeatureCompressor(model_path='path/to/ae_fused_256.pth')
        compressed = compressor.compress(features)  # (256, H, W)
    """
    
    def __init__(self, 
                 model_path=None,
                 feature_type='fused',
                 device='cuda'):
        """
        初始化压缩器
        
        Args:
            model_path: AutoEncoder模型路径
            feature_type: 特征类型 ('fused', 'dino', 'sd_s3' 等)
            device: 计算设备
        """
        self.device = device
        self.feature_type = feature_type
        
        # 获取配置
        config = FEATURE_CONFIGS[feature_type]
        self.input_dim = config['input_dim']
        self.output_dim = config['encoder_hidden_dims'][-1]  # 压缩后维度
        
        # 创建模型
        self.model = AutoencoderFlexible(
            input_dim=config['input_dim'],
            encoder_hidden_dims=config['encoder_hidden_dims'],
            decoder_hidden_dims=config['decoder_hidden_dims']
        ).to(device)
        
        # 加载权重
        if model_path is None:
            # 默认路径
            default_path = Path(__file__).parent.parent / f'dataset/room_0/Sequence_1/ae_models/ae_{feature_type}_256.pth'
            if default_path.exists():
                model_path = str(default_path)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"[FeatureCompressor] 加载模型: {model_path}")
            print(f"  压缩: {self.input_dim}维 → {self.output_dim}维")
        else:
            raise FileNotFoundError(f"未找到压缩模型: {model_path}")
    
    @torch.no_grad()
    def compress(self, features, return_numpy=False):
        """
        压缩特征
        
        Args:
            features: (C, H, W) 或 (B, C, H, W) 特征张量
            return_numpy: 是否返回numpy数组
            
        Returns:
            compressed: 压缩后的特征
        """
        # 确保输入是tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        
        # 处理维度
        squeeze_batch = False
        if features.dim() == 3:
            features = features.unsqueeze(0)
            squeeze_batch = True
        
        B, C, H, W = features.shape
        
        # 重排为 (B*H*W, C)
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C).to(self.device)
        
        # 编码
        compressed_flat = self.model.encode(features_flat)
        
        # 重排回 (B, C', H, W)
        C_out = compressed_flat.shape[-1]
        compressed = compressed_flat.reshape(B, H, W, C_out).permute(0, 3, 1, 2)
        
        if squeeze_batch:
            compressed = compressed.squeeze(0)
        
        if return_numpy:
            return compressed.cpu().numpy()
        return compressed.cpu()
    
    @torch.no_grad()
    def decompress(self, compressed):
        """
        解压缩特征 (用于验证)
        
        Args:
            compressed: (C', H, W) 压缩特征
            
        Returns:
            reconstructed: (C, H, W) 重建特征
        """
        if isinstance(compressed, np.ndarray):
            compressed = torch.from_numpy(compressed)
        
        squeeze_batch = False
        if compressed.dim() == 3:
            compressed = compressed.unsqueeze(0)
            squeeze_batch = True
        
        B, C, H, W = compressed.shape
        
        compressed_flat = compressed.permute(0, 2, 3, 1).reshape(-1, C).to(self.device)
        reconstructed_flat = self.model.decode(compressed_flat)
        
        C_out = reconstructed_flat.shape[-1]
        reconstructed = reconstructed_flat.reshape(B, H, W, C_out).permute(0, 3, 1, 2)
        
        if squeeze_batch:
            reconstructed = reconstructed.squeeze(0)
        
        return reconstructed.cpu()


def create_compressor(model_path=None, device='cuda'):
    """
    创建压缩器的便捷函数
    """
    return FeatureCompressor(model_path=model_path, device=device)
