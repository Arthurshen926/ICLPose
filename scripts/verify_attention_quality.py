#!/usr/bin/env python3
"""
验证Attention质量的诊断脚本
加载训练好的模型，分析attention heatmap的分布特性
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml

def analyze_attention_quality(checkpoint_path, config_path):
    """分析attention heatmap的质量"""
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 检查是否有保存的attention信息
    if 'attention_stats' in checkpoint:
        stats = checkpoint['attention_stats']
        print("=== Saved Attention Statistics ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    
    print(f"\n=== Model Checkpoint Analysis ===")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    # 分析注意力相关参数
    model_state = checkpoint.get('model', checkpoint.get('model_state_dict', {}))
    
    # 检查query embeddings
    if 'query_embed' in model_state:
        query_embed = model_state['query_embed']
        print(f"\n=== Query Embeddings ===")
        print(f"  Shape: {query_embed.shape}")
        print(f"  Mean: {query_embed.mean():.4f}")
        print(f"  Std: {query_embed.std():.4f}")
        print(f"  Min: {query_embed.min():.4f}")
        print(f"  Max: {query_embed.max():.4f}")

def compute_theoretical_attention_stats(n_points=1024, temperature=8.0, feature_dim=256):
    """计算理论上的attention分布特性"""
    print(f"\n=== Theoretical Attention Analysis ===")
    print(f"N_points: {n_points}")
    print(f"Temperature: {temperature}")
    print(f"Feature dim: {feature_dim}")
    
    # 随机初始化的query和tokens (模拟训练初期)
    torch.manual_seed(42)
    query = torch.randn(1, 64, feature_dim)  # (B, N_query, C)
    tokens = torch.randn(1, n_points, feature_dim)  # (B, N_points, C)
    
    # 计算attention scores
    scores = torch.matmul(query, tokens.transpose(1, 2)) / temperature
    
    # Softmax
    attn = torch.nn.functional.softmax(scores, dim=-1)
    
    print(f"\n--- Random Initialization (未训练) ---")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Attention max: {attn.max():.6f}")
    print(f"Attention min: {attn.min():.6f}")
    print(f"Attention std: {attn.std():.6f}")
    print(f"Expected uniform: {1/n_points:.6f}")
    
    # 模拟训练后的情况 (部分query对特定token有高关注)
    # 假设某些query-token对有更高的相似度
    scores_trained = scores.clone()
    for q in range(64):
        # 每个query关注约10个特定的points
        top_k = 10
        top_indices = torch.randint(0, n_points, (top_k,))
        scores_trained[0, q, top_indices] += 5.0  # 增加这些点的得分
    
    attn_trained = torch.nn.functional.softmax(scores_trained, dim=-1)
    
    print(f"\n--- Simulated After Training ---")
    print(f"Attention max: {attn_trained.max():.6f}")
    print(f"Attention min: {attn_trained.min():.6f}")
    print(f"Attention std: {attn_trained.std():.6f}")
    
    # 计算有效关注点数量 (熵)
    entropy = -(attn_trained * torch.log(attn_trained + 1e-10)).sum(dim=-1).mean()
    effective_n = torch.exp(entropy)
    print(f"Effective attention span: {effective_n:.1f} points (out of {n_points})")
    
    return attn, attn_trained

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--config', type=str, default=None, help='Config path')
    parser.add_argument('--theoretical', action='store_true', help='Run theoretical analysis')
    args = parser.parse_args()
    
    if args.theoretical:
        print("Running theoretical attention analysis...")
        attn_init, attn_trained = compute_theoretical_attention_stats()
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Query 0 的 attention分布
        axes[0].bar(range(100), attn_init[0, 0, :100].numpy())
        axes[0].set_title('Random Init Attention (Query 0, first 100 points)')
        axes[0].set_xlabel('Point index')
        axes[0].set_ylabel('Attention weight')
        
        axes[1].bar(range(100), attn_trained[0, 0, :100].numpy())
        axes[1].set_title('Simulated Trained Attention (Query 0, first 100 points)')
        axes[1].set_xlabel('Point index')
        axes[1].set_ylabel('Attention weight')
        
        plt.tight_layout()
        plt.savefig('attention_analysis.png', dpi=150)
        print("\nSaved attention_analysis.png")
        plt.show()
    
    if args.checkpoint and args.config:
        analyze_attention_quality(args.checkpoint, args.config)
