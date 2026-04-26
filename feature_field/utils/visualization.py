"""
可视化工具模块
用于训练过程中的定性评估
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def visualize_attention_maps(attention_weights, img, save_path, num_queries=8):
    """
    可视化注意力权重热力图（Query到2D特征的关注区域）
    
    Args:
        attention_weights: (B, N_query, N_img) 注意力权重
        img: (H, W, 3) RGB图像 [0, 1]
        save_path: 保存路径
        num_queries: 可视化前N个query
    """
    B, N_query, N_img = attention_weights.shape
    H, W = img.shape[:2]
    
    # 只可视化第一个batch
    attn = attention_weights[0].detach().cpu().numpy()  # (N_query, N_img)
    
    # 假设特征图是下采样的，需要reshape
    # 例如：N_img = (H//8) * (W//8) = 60 * 80
    feat_h, feat_w = H // 8, W // 8
    if N_img != feat_h * feat_w:
        # 尝试找到合适的分辨率
        feat_h = int(np.sqrt(N_img * H / W))
        feat_w = N_img // feat_h
    
    # 创建子图
    num_queries = min(num_queries, N_query)
    fig, axes = plt.subplots(2, num_queries // 2, figsize=(16, 6))
    axes = axes.flatten()
    
    for i in range(num_queries):
        # 获取第i个query的注意力
        attn_map = attn[i].reshape(feat_h, feat_w)
        
        # 上采样到原图大小
        attn_map_resized = cv2.resize(attn_map, (W, H))
        
        # 叠加到原图
        axes[i].imshow(img)
        axes[i].imshow(attn_map_resized, alpha=0.6, cmap='jet')
        axes[i].set_title(f'Query {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_2d3d_correspondence(img_feats, pcd_feats, img, pcd_xyz, save_path, num_samples=50):
    """
    可视化2D-3D特征相似度（隐式对应关系）
    
    Args:
        img_feats: (N_img, C) 2D特征 (numpy array或torch tensor)
        pcd_feats: (N_pcd, C) 3D特征 (numpy array或torch tensor)
        img: (H, W, 3) RGB图像
        pcd_xyz: (N_pcd, 3) 3D点云坐标 (numpy array或torch tensor)
        save_path: 保存路径
        num_samples: 采样多少个2D点进行可视化
    """
    # 转换为torch tensor并归一化
    if not torch.is_tensor(img_feats):
        img_feats = torch.from_numpy(img_feats).float()
    else:
        img_feats = img_feats.detach().cpu().float()
    
    if not torch.is_tensor(pcd_feats):
        pcd_feats = torch.from_numpy(pcd_feats).float()
    else:
        pcd_feats = pcd_feats.detach().cpu().float()
    
    if not isinstance(pcd_xyz, np.ndarray):
        pcd_xyz = pcd_xyz.detach().cpu().numpy()
    
    N_img, C = img_feats.shape
    N_pcd, _ = pcd_feats.shape
    H, W = img.shape[:2]
    
    # 归一化特征
    img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
    pcd_feats = torch.nn.functional.normalize(pcd_feats, dim=-1)
    
    # 计算相似度矩阵
    similarity = torch.mm(img_feats, pcd_feats.t())  # (N_img, N_pcd)
    
    # 随机采样一些2D点
    sample_indices = np.random.choice(N_img, min(num_samples, N_img), replace=False)
    
    # 为每个2D点找到最相似的3D点
    fig = plt.figure(figsize=(15, 5))
    
    # 左图：显示采样的2D点
    ax1 = fig.add_subplot(131)
    ax1.imshow(img)
    
    # 尝试推断特征图大小（假设与图像成比例）
    feat_h, feat_w = H // 8, W // 8
    if N_img != feat_h * feat_w:
        # 如果不匹配，尝试找到最接近的正方形或矩形
        ratio = W / H
        feat_h = int(np.sqrt(N_img / ratio))
        feat_w = N_img // feat_h
        # 如果还不匹配，使用近似值
        if feat_h * feat_w != N_img:
            feat_h = int(np.sqrt(N_img))
            feat_w = N_img // feat_h
    
    for idx in sample_indices[:10]:  # 只显示前10个
        if feat_h * feat_w == N_img:
            y, x = idx // feat_w, idx % feat_w
            # 映射到原图
            scale_y, scale_x = H / feat_h, W / feat_w
            y, x = int(y * scale_y + scale_y / 2), int(x * scale_x + scale_x / 2)
            ax1.plot(x, y, 'r+', markersize=10, markeredgewidth=2)
    
    ax1.set_title('Sampled 2D Points')
    ax1.axis('off')
    
    # 中图：相似度热力图
    ax2 = fig.add_subplot(132)
    sim_mean = similarity.mean(dim=1)
    
    # 只有当可以完美reshape时才显示热力图，否则显示柱状图
    if feat_h * feat_w == N_img:
        sim_mean_img = sim_mean.reshape(feat_h, feat_w).numpy()
        im = ax2.imshow(sim_mean_img, cmap='hot')
        plt.colorbar(im, ax=ax2)
    else:
        # 如果无法reshape，显示柱状图
        ax2.bar(range(len(sim_mean)), sim_mean.numpy())
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Similarity')
    
    ax2.set_title('Average 2D-3D Similarity')
    
    # 右图：3D点云（按相似度着色）
    ax3 = fig.add_subplot(133, projection='3d')
    
    # 计算每个3D点的平均相似度
    pcd_sim = similarity.mean(dim=0).numpy()
    
    # 降采样显示
    display_step = max(1, N_pcd // 2000)
    ax3.scatter(pcd_xyz[::display_step, 0], 
                pcd_xyz[::display_step, 1], 
                pcd_xyz[::display_step, 2],
                c=pcd_sim[::display_step], 
                cmap='viridis', 
                s=1)
    ax3.set_title('3D Points (colored by similarity)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_pose_prediction(gt_pose, pred_pose, img, save_path):
    """
    可视化位姿预测结果
    
    Args:
        gt_pose: (4, 4) GT位姿矩阵 (numpy array或torch tensor)
        pred_pose: (4, 4) 预测位姿矩阵 (numpy array或torch tensor)
        img: (H, W, 3) RGB图像
        save_path: 保存路径
    """
    # 转换为numpy数组
    if torch.is_tensor(gt_pose):
        gt_pose = gt_pose.detach().cpu().numpy()
    if torch.is_tensor(pred_pose):
        pred_pose = pred_pose.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 左图：原图
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # 中图：旋转误差可视化
    R_gt = gt_pose[:3, :3]
    R_pred = pred_pose[:3, :3]
    R_error = np.dot(R_pred, R_gt.T)
    
    # 计算旋转角度误差
    trace = np.trace(R_error)
    angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi
    
    axes[1].text(0.5, 0.5, f'Rotation Error:\n{angle_error:.2f}°', 
                 ha='center', va='center', fontsize=20,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    axes[1].set_title('Rotation Error')
    
    # 右图：平移误差可视化
    t_gt = gt_pose[:3, 3]
    t_pred = pred_pose[:3, 3]
    t_error = np.linalg.norm(t_pred - t_gt)
    
    axes[2].text(0.5, 0.7, f'Translation Error:\n{t_error:.3f}m', 
                 ha='center', va='center', fontsize=20,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 显示GT和预测的平移向量
    info_text = f'GT:   [{t_gt[0]:.2f}, {t_gt[1]:.2f}, {t_gt[2]:.2f}]\n'
    info_text += f'Pred: [{t_pred[0]:.2f}, {t_pred[1]:.2f}, {t_pred[2]:.2f}]'
    axes[2].text(0.5, 0.3, info_text, 
                 ha='center', va='center', fontsize=12,
                 family='monospace')
    
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')
    axes[2].set_title('Translation Error')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_query_features(query_list, save_path, num_layers=4):
    """
    可视化Query特征在不同融合层的演化
    
    Args:
        query_list: List[(B, N_query, C)] 每层的query特征
        save_path: 保存路径
        num_layers: 显示多少层
    """
    num_layers = min(num_layers, len(query_list))
    
    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))
    if num_layers == 1:
        axes = [axes]
    
    for i, queries in enumerate(query_list[:num_layers]):
        # 只可视化第一个batch
        q = queries[0].detach().cpu().numpy()  # (N_query, C)
        
        # PCA降维到2D进行可视化
        from sklearn.decomposition import PCA
        if q.shape[1] > 2:
            pca = PCA(n_components=2)
            q_2d = pca.fit_transform(q)
        else:
            q_2d = q[:, :2]
        
        # 散点图
        axes[i].scatter(q_2d[:, 0], q_2d[:, 1], alpha=0.6, s=50)
        axes[i].set_title(f'Layer {i+1}')
        axes[i].set_xlabel('PC1')
        axes[i].set_ylabel('PC2')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_feature_similarity_matrix(img_feats, pcd_feats, save_path):
    """
    可视化2D-3D特征相似度矩阵
    
    Args:
        img_feats: (N_img, C) 2D特征 (numpy array或torch tensor)
        pcd_feats: (N_pcd, C) 3D特征 (numpy array或torch tensor)
        save_path: 保存路径
    """
    # 转换为torch tensor
    if not torch.is_tensor(img_feats):
        img_feats = torch.from_numpy(img_feats).float()
    else:
        img_feats = img_feats.detach().cpu().float()
    
    if not torch.is_tensor(pcd_feats):
        pcd_feats = torch.from_numpy(pcd_feats).float()
    else:
        pcd_feats = pcd_feats.detach().cpu().float()
    
    # 归一化
    img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
    pcd_feats = torch.nn.functional.normalize(pcd_feats, dim=-1)
    
    # 相似度矩阵
    similarity = torch.mm(img_feats, pcd_feats.t()).numpy()  # (N_img, N_pcd)
    
    # 降采样显示（太大的矩阵显示不清）
    max_display = 500
    if similarity.shape[0] > max_display:
        step_img = similarity.shape[0] // max_display
        similarity = similarity[::step_img, :]
    if similarity.shape[1] > max_display:
        step_pcd = similarity.shape[1] // max_display
        similarity = similarity[:, ::step_pcd]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity, cmap='hot', aspect='auto')
    ax.set_xlabel('3D Points')
    ax.set_ylabel('2D Points')
    ax.set_title('2D-3D Feature Similarity Matrix')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
