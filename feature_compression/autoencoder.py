"""
共享的配置和模型定义
"""
import torch
import torch.nn as nn


# 针对不同特征类型的配置
FEATURE_CONFIGS = {
    'dino': {
        'input_dim': 768,
        'encoder_hidden_dims': [512, 256, 128, 64, 16],  # 768 -> 16维
        'decoder_hidden_dims': [64, 128, 256, 512, 768],
    },
    # 'fused': {
    #     'input_dim': 768,
    #     'encoder_hidden_dims': [512, 256, 128, 64, 16],  # 768 -> 16维 (与dino相同)
    #     'decoder_hidden_dims': [64, 128, 256, 512, 768],
    # },
    'fused': {
        'input_dim': 768,
        'encoder_hidden_dims': [512, 384, 256],  # 768 -> 256维
        'decoder_hidden_dims': [384, 512, 768],
    },
    'sd_s3': {
        'input_dim': 640,
        'encoder_hidden_dims': [512, 256, 128, 64, 16],  # 640 -> 16维
        'decoder_hidden_dims': [64, 128, 256, 512, 640],
    },
    'sd_s4': {
        'input_dim': 1280,
        'encoder_hidden_dims': [1024, 512, 256, 128, 32],  # 1280 -> 32维
        'decoder_hidden_dims': [128, 256, 512, 1024, 1280],
    },
    'sd_s5': {
        'input_dim': 1280,
        'encoder_hidden_dims': [1024, 512, 256, 128, 32],  # 1280 -> 32维
        'decoder_hidden_dims': [128, 256, 512, 1024, 1280],
    },
    # ── v2 多尺度特征金字塔配置 ──
    # 注: ODISE backbone 将所有 SD 特征投影到 512 维 (非原始 640/1280)
    # Fine 层: SD s3 和 DINO Patch 各自独立压缩, 压缩后拼接嵌入 3DGS
    'v2_fine_sd': {
        'input_dim': 512,
        'encoder_hidden_dims': [256, 128, 64],   # 512 -> 64维
        'decoder_hidden_dims': [128, 256, 512],
    },
    'v2_fine_dino': {
        'input_dim': 768,
        'encoder_hidden_dims': [384, 256, 64],   # 768 -> 64维
        'decoder_hidden_dims': [256, 384, 768],
    },
    'v2_mid': {
        'input_dim': 512,
        'encoder_hidden_dims': [256, 128, 64],   # 512 -> 64维
        'decoder_hidden_dims': [128, 256, 512],
    },
    'v2_coarse': {
        'input_dim': 512,
        'encoder_hidden_dims': [256, 128, 32],   # 512 -> 32维
        'decoder_hidden_dims': [128, 256, 512],
    },
}


class AutoencoderFlexible(nn.Module):
    """支持不同输入维度的Autoencoder
    
    改进 v2:
    - encode() 使用注册的 running min/max 代替 per-batch min-max
    - forward() 与 encode() 共享相同的归一化逻辑, 保证训练/推理一致
    - 支持 calibrate() 方法: 用训练数据统计全局 min/max
    """
    def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims):
        super(AutoencoderFlexible, self).__init__()
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(input_dim, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)
             
        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)

        # 全局 min/max 缓冲区 (通过 calibrate() 设置)
        bottleneck_dim = encoder_hidden_dims[-1]
        self.register_buffer('bottleneck_min', torch.zeros(bottleneck_dim))
        self.register_buffer('bottleneck_max', torch.ones(bottleneck_dim))
        self.register_buffer('is_calibrated', torch.tensor(False))

        # 输入归一化缓冲区 (通过 set_input_norm() 设置)
        # 消除不同特征尺度间的数量级差异 (例如 SD fine 均值~9.5 vs coarse 均值~0.7)
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('has_input_norm', torch.tensor(False))

    def set_input_norm(self, mean: torch.Tensor, std: torch.Tensor):
        """设置输入归一化参数 (per-channel z-score).

        Args:
            mean: [input_dim] 每通道均值
            std:  [input_dim] 每通道标准差 (接近 0 的通道被截断到 1e-6)
        """
        self.input_mean.copy_(mean.to(self.input_mean.device))
        self.input_std.copy_(std.clamp(min=1e-6).to(self.input_std.device))
        self.has_input_norm.fill_(True)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """若已设置归一化参数则应用 z-score 归一化, 否则原样返回."""
        if self.has_input_norm:
            return (x - self.input_mean) / self.input_std
        return x

    def _denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """与 _normalize_input 互逆, 将网络输出映射回原始特征空间."""
        if self.has_input_norm:
            return x * self.input_std + self.input_mean
        return x

    def _encode_raw(self, x):
        """编码器前向: 输入归一化 → encoder → L2 norm on bottleneck"""
        x = self._normalize_input(x)
        for m in self.encoder:
            x = m(x)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x

    def forward(self, x):
        """训练用: normalize → encode → L2 norm → decode → 返回归一化空间的重建结果.
        
        Loss 应在归一化空间计算: mse(forward(x), model._normalize_input(x))
        这样无论输入特征量级如何, MSE 都保持 O(1) 尺度.
        """
        z = self._encode_raw(x)
        out = z
        for m in self.decoder:
            out = m(out)
        return out

    def encode(self, x):
        """推理用: encode → L2 norm → min-max 归一化到 [0,1]
        
        若已 calibrate(), 使用全局 min/max;
        否则回退到 per-batch min/max (兼容旧代码)
        """
        z = self._encode_raw(x)
        if self.is_calibrated:
            z = (z - self.bottleneck_min) / (self.bottleneck_max - self.bottleneck_min + 1e-12)
        else:
            z = (z - torch.min(z)) / (torch.max(z) - torch.min(z) + 1e-12)
        return z.clamp(0, 1)

    def decode(self, x):
        for m in self.decoder:
            x = m(x)    
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x

    @torch.no_grad()
    def calibrate(self, data_loader, device='cuda', max_batches=50):
        """用训练数据统计 bottleneck 的全局 min/max
        
        Args:
            data_loader: 提供 (batch,) 的 DataLoader
            device: 计算设备
            max_batches: 最多统计多少个 batch
        """
        self.eval()
        global_min = None
        global_max = None
        for i, (batch,) in enumerate(data_loader):
            if i >= max_batches:
                break
            batch = batch.to(device)
            z = self._encode_raw(batch)
            batch_min = z.min(dim=0).values
            batch_max = z.max(dim=0).values
            if global_min is None:
                global_min = batch_min
                global_max = batch_max
            else:
                global_min = torch.min(global_min, batch_min)
                global_max = torch.max(global_max, batch_max)
        
        self.bottleneck_min.copy_(global_min)
        self.bottleneck_max.copy_(global_max)
        self.is_calibrated.fill_(True)
        print(f"    ✓ Calibrated: min=[{global_min.min():.4f}, {global_min.max():.4f}], "
              f"max=[{global_max.min():.4f}, {global_max.max():.4f}]")
