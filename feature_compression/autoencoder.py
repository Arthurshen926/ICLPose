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
    """支持不同输入维度的Autoencoder"""
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

    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def encode(self, x):
        for m in self.encoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-12)
        return x

    def decode(self, x):
        for m in self.decoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x
