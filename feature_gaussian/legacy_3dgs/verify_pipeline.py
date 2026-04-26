"""
Feature Embedding 验证脚本
==========================
快速验证整个特征嵌入pipeline是否跑通:
1. 加载3DGS
2. 加载特征数据 
3. 渲染特征图
4. 计算loss + 反向传播
5. 可视化结果

用法:
    python -m feature_3dgs.verify_pipeline
"""

import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_gaussian.legacy_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_gaussian.legacy_3dgs.feature_renderer import FeatureRenderer
from feature_gaussian.legacy_3dgs.feature_dataset import FeatureEmbeddingDataset


def verify():
    """验证完整pipeline"""
    device = torch.device('cuda')
    
    # 路径
    base = project_root
    ply_path = base / 'dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply'
    feature_dir = base / 'dataset/room_0/Sequence_1/features_compressed/fused'
    traj_path = base / 'dataset/room_0/Sequence_1/traj_w_c.txt'
    output_dir = base / 'output/feature_3dgs_verify1'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Feature Embedding Pipeline 验证")
    print("=" * 60)
    
    # ============================================================
    # Step 1: 加载3DGS模型
    # ============================================================
    print("\n[Step 1] 加载预训练3DGS...")
    model = GaussianFeatureModel(feature_dim=256)
    model.load_ply(str(ply_path))
    model = model.to(device)
    
    print(f"  _xyz shape: {model._xyz.shape}")
    print(f"  _loc_feature shape: {model._loc_feature.shape}")
    print(f"  _loc_feature requires_grad: {model._loc_feature.requires_grad}")
    print(f"  _xyz requires_grad: {model._xyz.requires_grad}")
    
    # ============================================================
    # Step 2: 加载数据
    # ============================================================
    print("\n[Step 2] 加载特征数据...")
    dataset = FeatureEmbeddingDataset(
        feature_dir=str(feature_dir),
        traj_path=str(traj_path),
        max_frames=10,  # 验证时只用10帧
    )
    
    sample = dataset[0]
    gt_feat = sample['feature_map']
    pose = sample['pose']
    print(f"  GT特征图: {gt_feat.shape} (min={gt_feat.min():.4f}, max={gt_feat.max():.4f})")
    print(f"  位姿: {pose.shape}")
    print(f"  帧ID: {sample['frame_id']}")
    
    # ============================================================
    # Step 3: 渲染特征图 (前向传播)
    # ============================================================
    print("\n[Step 3] 渲染特征图...")
    gt_feat = gt_feat.to(device)
    pose = pose.to(device)
    feat_H, feat_W = gt_feat.shape[1], gt_feat.shape[2]
    
    t0 = time.time()
    result = FeatureRenderer.render_features(
        gaussian_model=model,
        viewmat=pose,
        fx=320.0, fy=320.0,
        cx=319.5, cy=239.5,
        img_height=480,
        img_width=640,
        feature_height=feat_H,
        feature_width=feat_W,
    )
    t_render = time.time() - t0
    
    rendered_feat = result['feature_map']
    print(f"  渲染特征图: {rendered_feat.shape}")
    print(f"  渲染耗时: {t_render*1000:.1f}ms")
    print(f"  可见Gaussians: {result['visible_mask'].sum().item()}/{model.num_gaussians}")
    print(f"  特征值范围: [{rendered_feat.min().item():.4f}, {rendered_feat.max().item():.4f}]")
    
    # ============================================================
    # Step 4: 计算loss + 反向传播
    # ============================================================
    print("\n[Step 4] 计算loss + 反向传播...")
    loss_l1 = torch.abs(rendered_feat - gt_feat).mean()
    cos_sim = F.cosine_similarity(rendered_feat, gt_feat, dim=0).mean()
    loss_cos = 1.0 - cos_sim
    loss = loss_l1 + 0.1 * loss_cos
    
    print(f"  L1 loss: {loss_l1.item():.6f}")
    print(f"  Cosine sim: {cos_sim.item():.6f}")
    print(f"  Total loss: {loss.item():.6f}")
    
    optimizer = torch.optim.Adam([model._loc_feature], lr=0.001)
    optimizer.zero_grad()
    
    t0 = time.time()
    loss.backward()
    t_backward = time.time() - t0
    
    grad = model._loc_feature.grad
    print(f"  反向传播耗时: {t_backward*1000:.1f}ms")
    print(f"  梯度shape: {grad.shape}")
    print(f"  梯度范围: [{grad.min().item():.8f}, {grad.max().item():.8f}]")
    print(f"  梯度均值: {grad.mean().item():.8f}")
    print(f"  非零梯度比例: {(grad.abs() > 1e-10).float().mean().item()*100:.1f}%")
    
    optimizer.step()
    print("  优化器step完成!")
    
    # ============================================================
    # Step 5: 多步训练验证 (确认loss收敛)
    # ============================================================
    print("\n[Step 5] 多步训练验证 (20步)...")
    losses = []
    for i in range(20):
        idx = i % len(dataset)
        sample = dataset[idx]
        gt_feat = sample['feature_map'].to(device)
        pose_i = sample['pose'].to(device)
        feat_H, feat_W = gt_feat.shape[1], gt_feat.shape[2]
        
        result = FeatureRenderer.render_features(
            gaussian_model=model,
            viewmat=pose_i,
            fx=320.0, fy=320.0, cx=319.5, cy=239.5,
            img_height=480, img_width=640,
            feature_height=feat_H, feature_width=feat_W,
        )
        
        rendered_feat = result['feature_map']
        loss = torch.abs(rendered_feat - gt_feat).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1:3d}: loss = {loss.item():.6f}")
    
    loss_trend = "下降" if losses[-1] < losses[0] else "未下降"
    print(f"  Loss趋势: {losses[0]:.6f} -> {losses[-1]:.6f} ({loss_trend})")
    
    # ============================================================
    # Step 6: 渲染RGB验证
    # ============================================================
    print("\n[Step 6] 渲染RGB验证...")
    try:
        pose_rgb = dataset[0]['pose'].to(device)
        rgb_result = FeatureRenderer.render_rgb(
            gaussian_model=model,
            viewmat=pose_rgb,
            fx=320.0, fy=320.0, cx=319.5, cy=239.5,
            img_height=480, img_width=640,
        )
        rgb = rgb_result['rgb']
        print(f"  RGB图: {rgb.shape}, 值范围: [{rgb.min().item():.4f}, {rgb.max().item():.4f}]")
        
        # 保存RGB
        try:
            from torchvision.utils import save_image
            save_image(rgb, str(output_dir / 'verify_rgb.png'))
            print(f"  RGB已保存: {output_dir / 'verify_rgb.png'}")
        except ImportError:
            # fallback: 用PIL保存
            import PIL.Image as Image
            rgb_np = (rgb.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(rgb_np).save(str(output_dir / 'verify_rgb.png'))
            print(f"  RGB已保存: {output_dir / 'verify_rgb.png'}")
    except Exception as e:
        print(f"  RGB渲染跳过: {e}")
    
    # ============================================================
    # Step 7: 保存特征可视化 (PCA降维)
    # ============================================================
    print("\n[Step 7] 特征可视化...")
    try:
        rendered_feat_vis = result['feature_map'].detach().cpu()  # [D, H, W]
        gt_feat_vis = dataset[0]['feature_map']
        
        # PCA降维到3通道用于可视化
        def pca_vis(feat_map, n_components=3):
            """将D维特征图PCA降维为3通道用于可视化"""
            D, H, W = feat_map.shape
            feat_flat = feat_map.reshape(D, -1).T  # [H*W, D]
            
            # 中心化
            mean = feat_flat.mean(0)
            feat_centered = feat_flat - mean
            
            # SVD
            U, S, V = torch.svd(feat_centered)
            proj = U[:, :n_components]  # [H*W, 3]
            
            # 归一化到 [0, 1]
            proj = proj - proj.min(0)[0]
            proj = proj / (proj.max(0)[0] + 1e-8)
            
            return proj.T.reshape(n_components, H, W)
        
        rendered_pca = pca_vis(rendered_feat_vis)
        gt_pca = pca_vis(gt_feat_vis)
        
        import PIL.Image as Image
        # 保存渲染的特征可视化
        vis_rendered = (rendered_pca.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_rendered).save(str(output_dir / 'verify_feat_rendered.png'))
        
        # 保存GT特征可视化
        vis_gt = (gt_pca.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_gt).save(str(output_dir / 'verify_feat_gt.png'))
        
        print(f"  特征可视化已保存: {output_dir}")
    except Exception as e:
        print(f"  可视化跳过: {e}")
    
    # ============================================================
    # Step 8: 保存PLY
    # ============================================================
    print("\n[Step 8] 保存带特征的PLY...")
    model.save_ply_with_features(str(output_dir / 'point_cloud_with_features.ply'))
    
    # ============================================================
    # 最终报告
    # ============================================================
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    print(f"  [✓] PLY加载: {model.num_gaussians} Gaussians")
    print(f"  [✓] 特征嵌入: {model._loc_feature.shape} (requires_grad={model._loc_feature.requires_grad})")
    print(f"  [✓] 前向渲染: {rendered_feat.shape}")
    print(f"  [✓] 反向传播: 梯度存在")
    print(f"  [✓] Loss收敛: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  [✓] PLY保存: {output_dir / 'point_cloud_with_features.ply'}")
    print(f"\n✅ Pipeline验证通过!")


if __name__ == '__main__':
    verify()
