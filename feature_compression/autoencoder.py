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
    # Fine 层: SD s3 和 DINO Patch 各自独立压缩, 压缩后拼接嵌入 3DGS
    'v2_fine_sd': {
        'input_dim': 640,
        'encoder_hidden_dims': [384, 256, 64],   # 640 -> 64维
        'decoder_hidden_dims': [256, 384, 640],
    },
    'v2_fine_dino': {
        'input_dim': 768,
        'encoder_hidden_dims': [384, 256, 64],   # 768 -> 64维
        'decoder_hidden_dims': [256, 384, 768],
    },
    'v2_mid': {
        'input_dim': 1280,
        'encoder_hidden_dims': [512, 256, 64],   # 1280 -> 64维
        'decoder_hidden_dims': [256, 512, 1280],
    },
    'v2_coarse': {
        'input_dim': 1280,
        'encoder_hidden_dims': [512, 256, 32],   # 1280 -> 32维
        'decoder_hidden_dims': [256, 512, 1280],
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

    def _encode_raw(self, x):
        """编码器前向, 仅 L2 归一化, 不做 min-max"""
        for m in self.encoder:
            x = m(x)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x

    def forward(self, x):
        """训练用: encode → L2 norm → decode (不做 min-max, 保持与 decode 对齐)"""
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
