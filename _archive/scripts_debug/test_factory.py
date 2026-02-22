#!/usr/bin/env python3
"""测试模型和损失工厂"""

import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

import yaml
import torch

# 加载配置
with open('configs/exp020_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('配置加载成功！')
print(f'  模型版本: {config["model"]["version"]}')
print(f'  损失版本: {config["loss"]["version"]}')
print(f'  Transformer层数: {config["model"]["num_layers"]}')
print(f'  重叠检测: {config["model"]["use_overlap_detection"]}')
print(f'  C2F: {config["model"]["use_c2f"]}')

# 测试模型工厂
from utils.model_factory import create_model, print_model_summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(config, device)
print('')
print_model_summary(model)

# 测试损失工厂
from utils.loss_factory import create_combined_loss
criterion = create_combined_loss(config, device)
print('')
print(f'损失函数类型: {type(criterion).__name__}')
print(f'  - 重叠损失: {criterion.use_overlap_loss}')
print(f'  - Diversity损失: {criterion.use_diversity_loss}')
print(f'  - 重投影损失: {criterion.use_reprojection_loss}')

print('\n✅ 所有测试通过！')
