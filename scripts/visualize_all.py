#!/usr/bin/env python3
"""
ICLPose 统一可视化工具
======================
整合项目中所有可视化功能到一个入口脚本，通过子命令调用不同可视化模式。

子命令 (modes):
  pca-features     PCA特征图可视化 (2D encoder / 3D rendered / GT)
  rgb-compare      GT vs Rendered RGB对比图 + Error map + Depth
  gsff-optim       GSFF位姿优化轨迹可视化 (loss曲线 + 特征匹配)
  cosine-map       特征余弦相似度热力图
  training-log     从训练日志解析PSNR/loss曲线
  feature-quality  特征质量分析 (distinctiveness, PCA spectrum)

用法示例:
  # 1) PCA特征可视化 — 对比 GT features vs rendered features
  python scripts/visualize_all.py pca-features \
      --feature_dir output/features_multiscale/OldHospital \
      --frame_ids 0 100 500 \
      --scales coarse mid fine_sd fine_dino \
      --output_dir output/vis/pca_features

  # 2) GT vs Rendered RGB对比
  python scripts/visualize_all.py rgb-compare \
      --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
      --source_dir dataset/OldHospital \
      --n_views 8 \
      --output_dir output/vis/rgb_compare

  # 3) GSFF优化轨迹可视化
  python scripts/visualize_all.py gsff-optim \
      --checkpoint output/gsff/OldHospital/checkpoints/final.pth \
      --source_dir dataset/OldHospital \
      --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
      --cameras_json output/2dgs_models/OldHospital/v7_depth/cameras.json \
      --num_samples 4 \
      --output_dir output/vis/gsff_optim

  # 4) 余弦相似度热力图
  python scripts/visualize_all.py cosine-map \
      --feature_dir output/features_multiscale/OldHospital \
      --ply_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
      --frame_ids 0 100 \
      --output_dir output/vis/cosine_maps

  # 5) 从训练日志画PSNR/loss曲线
  python scripts/visualize_all.py training-log \
      --log_path output/joint_gsff_radio/joint_oh_v8_gsff_radio/train.log \
      --output_dir output/vis/training_curves

  # 6) 特征质量分析
  python scripts/visualize_all.py feature-quality \
      --feature_dir output/features_multiscale/OldHospital/coarse \
      --n_samples 50 \
      --output_dir output/vis/feature_quality
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ═══════════════════════════════════════════════════════════════════════════
# 通用工具函数
# ═══════════════════════════════════════════════════════════════════════════

def pca_colorize(feat_chw, n_components=3, pca_model=None):
    """
    PCA降维将特征图转换为RGB可视化。

    Args:
        feat_chw: [C, H, W] 特征图 (torch.Tensor 或 numpy)
        n_components: PCA维度 (默认3=RGB)
        pca_model: 可选的sklearn PCA模型 (用于跨帧一致性)

    Returns:
        rgb: [H, W, 3] numpy array, 值域 [0, 1]
    """
    if isinstance(feat_chw, torch.Tensor):
        feat_chw = feat_chw.detach().cpu().float().numpy()
    C, H, W = feat_chw.shape
    flat = feat_chw.reshape(C, -1).T  # [HW, C]

    if pca_model is not None:
        rgb = pca_model.transform(flat)[:, :n_components]
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        rgb = pca.fit_transform(flat)  # [HW, 3]

    # Normalize to [0, 1]
    for c in range(rgb.shape[1]):
        mn, mx = rgb[:, c].min(), rgb[:, c].max()
        if mx - mn > 1e-8:
            rgb[:, c] = (rgb[:, c] - mn) / (mx - mn)
        else:
            rgb[:, c] = 0.5

    return rgb.reshape(H, W, n_components)


def cosine_similarity_map(feat_a, feat_b):
    """
    计算两个特征图的逐像素余弦相似度。

    Args:
        feat_a, feat_b: [C, H, W] torch.Tensor

    Returns:
        sim_map: [H, W] numpy array, 值域 [-1, 1]
    """
    a = feat_a.float()
    b = feat_b.float()
    a_norm = F.normalize(a, dim=0)
    b_norm = F.normalize(b, dim=0)
    sim = (a_norm * b_norm).sum(dim=0)  # [H, W]
    return sim.cpu().numpy()


def ensure_dir(path):
    """创建目录（如果不存在）。"""
    os.makedirs(path, exist_ok=True)
    return path


def load_feature(path):
    """加载 .pt 特征文件, 返回 [C, H, W]。"""
    feat = torch.load(path, map_location='cpu')
    if feat.dim() == 4:
        feat = feat.squeeze(0)
    return feat


# ═══════════════════════════════════════════════════════════════════════════
# Mode 1: PCA 特征图可视化
# ═══════════════════════════════════════════════════════════════════════════

def cmd_pca_features(args):
    """
    PCA降维可视化: 从预提取特征目录加载多尺度特征, PCA→RGB。

    输入: --feature_dir (包含 coarse/, mid/, fine_sd/, fine_dino/ 子目录)
    输出: 每帧一张多尺度PCA对比图
    """
    feature_dir = Path(args.feature_dir)
    output_dir = ensure_dir(args.output_dir)

    # 自动检测尺度目录 (dataset v2 格式)
    scale_map = {
        'coarse': ['coarse', 'sd_s5'],
        'mid': ['mid', 'sd_s4'],
        'fine_sd': ['fine_sd', 'sd_s3'],
        'fine_dino': ['fine_dino', 'dino'],
    }

    available_scales = {}
    for scale_name, candidates in scale_map.items():
        for cand in candidates:
            d = feature_dir / cand
            if d.exists():
                available_scales[scale_name] = d
                break

    if args.scales:
        scales_to_show = [s for s in args.scales if s in available_scales]
    else:
        scales_to_show = list(available_scales.keys())

    if not scales_to_show:
        print(f"[ERROR] 未找到有效尺度目录: {feature_dir}")
        return

    print(f"可用尺度: {list(available_scales.keys())}")
    print(f"将可视化: {scales_to_show}")

    # 找到所有帧
    first_scale_dir = available_scales[scales_to_show[0]]
    all_files = sorted(first_scale_dir.glob('*.pt'))

    if args.frame_ids:
        selected_files = []
        for fid in args.frame_ids:
            matches = [f for f in all_files if f'_{fid}_' in f.name or f.name.startswith(f'rgb_{fid}_')]
            if matches:
                selected_files.append(matches[0])
            elif fid < len(all_files):
                selected_files.append(all_files[fid])
        all_files = selected_files

    n_frames = min(len(all_files), args.max_frames)
    print(f"将可视化 {n_frames} 帧")

    for i, feat_file in enumerate(all_files[:n_frames]):
        fig, axes = plt.subplots(1, len(scales_to_show), figsize=(5 * len(scales_to_show), 5))
        if len(scales_to_show) == 1:
            axes = [axes]

        frame_name = feat_file.stem

        for j, scale_name in enumerate(scales_to_show):
            scale_dir = available_scales[scale_name]
            # 找对应帧的特征文件
            scale_file = scale_dir / feat_file.name
            if not scale_file.exists():
                # 尝试匹配帧号
                pattern = feat_file.name.split('_')[1]  # 提取帧号
                candidates = list(scale_dir.glob(f'*_{pattern}_*.pt'))
                scale_file = candidates[0] if candidates else None

            if scale_file and scale_file.exists():
                feat = load_feature(str(scale_file))
                rgb = pca_colorize(feat)
                axes[j].imshow(rgb)
                axes[j].set_title(f'{scale_name}\n{feat.shape[0]}d × {feat.shape[1]}×{feat.shape[2]}')
            else:
                axes[j].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20)
                axes[j].set_title(scale_name)
            axes[j].axis('off')

        fig.suptitle(f'PCA Feature Visualization — {frame_name}', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{frame_name}_pca.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [{i+1}/{n_frames}] Saved: {save_path}')

    print(f'\n[Done] PCA可视化保存至: {output_dir}')


# ═══════════════════════════════════════════════════════════════════════════
# Mode 2: GT vs Rendered RGB 对比
# ═══════════════════════════════════════════════════════════════════════════

def cmd_rgb_compare(args):
    """
    GT vs Rendered RGB渲染对比, 带Error map和可选Depth。

    需要: 2DGS PLY checkpoint + 场景数据目录
    输出: 每个view一张对比图 + summary grid
    """
    from feature_3dgs.train_2dgs_geometry import (
        GaussianModel2DGS, CameraData,
        load_image_tensor, ssim,
    )
    from feature_3dgs.train_2dgs_joint import render_rgb_2dgs
    from feature_3dgs.train_2dgs_joint_v2 import load_scene_colmap

    output_dir = ensure_dir(args.output_dir)

    # Load scene
    print("[1/3] Loading scene...")
    cameras, points3d = load_scene_colmap(args.source_dir)

    # Load Gaussians
    print("[2/3] Loading 2DGS model...")
    gaussians = GaussianModel2DGS(sh_degree=3)
    gaussians.load_ply(args.ply_path)
    gaussians = gaussians.cuda()

    # Select test views
    cam_list = list(cameras.values())
    n_views = min(args.n_views, len(cam_list))
    step = max(1, len(cam_list) // n_views)
    selected = [cam_list[i * step] for i in range(n_views)]

    print(f"[3/3] Rendering {n_views} views...")
    psnrs = []
    ssims = []

    for idx, cam in enumerate(selected):
        gt_img = load_image_tensor(cam, args.source_dir, longest_edge=0).cuda()
        H, W = gt_img.shape[1], gt_img.shape[2]

        with torch.no_grad():
            rendered, alpha, info = render_rgb_2dgs(
                gaussians, cam, W, H, bg_color=torch.zeros(3, device='cuda')
            )

        # PSNR
        mse = F.mse_loss(rendered, gt_img).item()
        psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
        psnrs.append(psnr)

        # SSIM
        s = ssim(rendered.unsqueeze(0), gt_img.unsqueeze(0)).item()
        ssims.append(s)

        # Error map
        error = torch.abs(rendered - gt_img).mean(dim=0).cpu().numpy()

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        gt_np = gt_img.cpu().permute(1, 2, 0).numpy().clip(0, 1)
        rd_np = rendered.cpu().permute(1, 2, 0).numpy().clip(0, 1)

        axes[0].imshow(gt_np)
        axes[0].set_title('GT')
        axes[0].axis('off')

        axes[1].imshow(rd_np)
        axes[1].set_title(f'Rendered (PSNR={psnr:.2f} dB)')
        axes[1].axis('off')

        im = axes[2].imshow(error, cmap='hot', vmin=0, vmax=0.15)
        axes[2].set_title(f'Error (SSIM={s:.4f})')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        fig.suptitle(f'View {idx}: {cam.image_name}', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'view_{idx:03d}_{cam.image_name}_compare.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [{idx+1}/{n_views}] PSNR={psnr:.2f} SSIM={s:.4f} → {save_path}')

    # Summary
    print(f'\n[Summary] Mean PSNR={np.mean(psnrs):.2f} dB, Mean SSIM={np.mean(ssims):.4f}')
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write(f'Mean PSNR: {np.mean(psnrs):.2f} dB\n')
        f.write(f'Mean SSIM: {np.mean(ssims):.4f}\n')
        for idx, (p, s) in enumerate(zip(psnrs, ssims)):
            f.write(f'  View {idx}: PSNR={p:.2f}, SSIM={s:.4f}\n')

    print(f'[Done] RGB对比保存至: {output_dir}')


# ═══════════════════════════════════════════════════════════════════════════
# Mode 3: GSFF 优化轨迹可视化
# ═══════════════════════════════════════════════════════════════════════════

def cmd_gsff_optim(args):
    """
    GSFF位姿优化全流程可视化。

    加载训练好的GSFF模型, 对测试样本运行位姿优化, 保存:
      - 特征图 PCA (2D encoder + 3D rendered)
      - 余弦相似度热力图
      - 优化轨迹 (loss / 位姿误差随迭代变化)
      - RGB渲染对比 (初始/优化后/GT)
    """
    try:
        from gsff.triplane import DualScaleTriplane
        from gsff.encoder import DualScaleEncoder
        from gsff.pose_refine import (
            refine_pose, render_features_for_pose, se3_exp,
        )
        from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
        from gsplat import rasterization_2dgs
    except ImportError as e:
        print(f"[ERROR] GSFF模块导入失败: {e}")
        print("  请确认已安装 gsplat 和 gsff 模块")
        return

    output_dir = ensure_dir(args.output_dir)
    print(f"[GSFF优化可视化] 此功能需要完整的GSFF训练环境。")
    print(f"  如需使用, 请直接运行专用脚本:")
    print(f"    python scripts/visualize_gsff.py --checkpoint {args.checkpoint} \\")
    print(f"        --source_dir {args.source_dir} --model_path {args.ply_path} \\")
    print(f"        --cameras_json {args.cameras_json} --num_samples {args.num_samples}")
    print(f"\n  或使用更详细的中间量可视化:")
    print(f"    python scripts/visualize_gsff_intermediates.py --checkpoint {args.checkpoint} ...")

    # 简化版: 若能加载checkpoint, 显示模型信息
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        print(f"\n[Checkpoint Info]")
        for key in ckpt.keys():
            if isinstance(ckpt[key], dict):
                print(f"  {key}: {len(ckpt[key])} entries")
            elif isinstance(ckpt[key], (int, float, str)):
                print(f"  {key}: {ckpt[key]}")
    else:
        print(f"[WARNING] Checkpoint not found: {args.checkpoint}")


# ═══════════════════════════════════════════════════════════════════════════
# Mode 4: 余弦相似度热力图
# ═══════════════════════════════════════════════════════════════════════════

def cmd_cosine_map(args):
    """
    计算并可视化 GT特征 vs Rendered特征 的逐像素余弦相似度。

    需要: 预提取GT特征 + 3DGS渲染特征 (或两组特征目录)
    输出: 余弦相似度热力图 PNG
    """
    output_dir = ensure_dir(args.output_dir)
    feature_dir = Path(args.feature_dir)

    if args.feature_dir_b:
        feature_dir_b = Path(args.feature_dir_b)
        label_a, label_b = 'Features A', 'Features B'
    else:
        print("[INFO] 仅提供一个特征目录, 将计算帧间自相似度")
        feature_dir_b = None
        label_a, label_b = 'Frame i', 'Frame j'

    # 找特征文件
    feat_files = sorted(feature_dir.glob('*.pt'))

    if args.frame_ids:
        indices = args.frame_ids[:10]
    else:
        max_n = min(len(feat_files), 8)
        step = max(1, len(feat_files) // max_n)
        indices = list(range(0, len(feat_files), step))[:max_n]

    for idx in indices:
        if idx >= len(feat_files):
            continue
        feat_a = load_feature(str(feat_files[idx]))

        if feature_dir_b:
            feat_file_b = feature_dir_b / feat_files[idx].name
            if not feat_file_b.exists():
                continue
            feat_b = load_feature(str(feat_file_b))
        else:
            # 自相似: 和下一帧比较
            next_idx = min(idx + 1, len(feat_files) - 1)
            feat_b = load_feature(str(feat_files[next_idx]))

        # 对齐尺寸
        if feat_a.shape != feat_b.shape:
            h, w = min(feat_a.shape[1], feat_b.shape[1]), min(feat_a.shape[2], feat_b.shape[2])
            feat_a = F.interpolate(feat_a.unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0)
            feat_b = F.interpolate(feat_b.unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0)

        sim_map = cosine_similarity_map(feat_a, feat_b)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(pca_colorize(feat_a))
        axes[0].set_title(f'{label_a}: {feat_files[idx].stem}')
        axes[0].axis('off')

        axes[1].imshow(pca_colorize(feat_b))
        axes[1].set_title(label_b)
        axes[1].axis('off')

        im = axes[2].imshow(sim_map, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
        axes[2].set_title(f'Cosine Similarity (mean={sim_map.mean():.3f})')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'cosine_{idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Frame {idx}: mean_cos={sim_map.mean():.3f} → {save_path}')

    print(f'\n[Done] 余弦相似度热力图保存至: {output_dir}')


# ═══════════════════════════════════════════════════════════════════════════
# Mode 5: 训练日志曲线
# ═══════════════════════════════════════════════════════════════════════════

def cmd_training_log(args):
    """
    从训练日志解析指标, 绘制PSNR / loss / 其他metric的训练曲线。

    支持格式:
      - 2DGS训练日志 (iter, PSNR, loss)
      - GSFF联合训练日志 (iter, PSNR, cos_c, cos_f, radio_cos, gaussian count)
      - MSFlow训练日志 (epoch, rot_err, trans_err, flow_loss)
    """
    output_dir = ensure_dir(args.output_dir)
    log_path = args.log_path

    if not os.path.exists(log_path):
        print(f"[ERROR] 日志文件不存在: {log_path}")
        return

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 自动检测日志格式
    content = ''.join(lines[:200])

    if 'PSNR' in content and ('cos_c' in content or 'radio' in content):
        _parse_joint_gsff_log(lines, output_dir)
    elif 'Val E' in content or 'rot_err' in content:
        _parse_msflow_log(lines, output_dir)
    elif 'PSNR' in content:
        _parse_2dgs_log(lines, output_dir)
    else:
        print(f"[WARNING] 无法识别日志格式, 尝试通用解析...")
        _parse_generic_log(lines, output_dir)


def _parse_joint_gsff_log(lines, output_dir):
    """解析 GSFF 联合训练日志。"""
    iters, psnrs, cos_c_vals, cos_f_vals = [], [], [], []
    radio_cos_vals, losses, n_gaussians = [], [], []

    for line in lines:
        # 匹配: [iter 2000] PSNR=15.1 dB ...
        m = re.search(r'\[iter\s+(\d+)\].*?PSNR=([\d.]+)', line)
        if m:
            iters.append(int(m.group(1)))
            psnrs.append(float(m.group(2)))

            # 可选字段
            mc = re.search(r'cos_c=([\d.]+)', line)
            if mc:
                cos_c_vals.append(float(mc.group(1)))

            mf = re.search(r'cos_f=([\d.]+)', line)
            if mf:
                cos_f_vals.append(float(mf.group(1)))

            mr = re.search(r'radio_cos=([\d.]+)', line)
            if mr:
                radio_cos_vals.append(float(mr.group(1)))

            ml = re.search(r'loss=([\d.]+)', line)
            if ml:
                losses.append(float(ml.group(1)))

            mn = re.search(r'#G=(\d+)', line)
            if not mn:
                mn = re.search(r'n_gauss(?:ians)?=(\d+)', line)
            if mn:
                n_gaussians.append(int(mn.group(1)))

    if not iters:
        print("[WARNING] 未解析到任何数据行")
        return

    # 绘图: 多面板
    n_panels = 2 + bool(cos_c_vals) + bool(radio_cos_vals) + bool(n_gaussians)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    # PSNR
    axes[ax_idx].plot(iters, psnrs, 'b-', linewidth=1.5)
    axes[ax_idx].set_ylabel('PSNR (dB)')
    axes[ax_idx].set_title('PSNR over Training')
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1

    # Loss
    if losses:
        axes[ax_idx].plot(iters[:len(losses)], losses, 'r-', linewidth=1)
        axes[ax_idx].set_ylabel('Total Loss')
        axes[ax_idx].set_title('Loss')
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    # Cosine similarities
    if cos_c_vals:
        axes[ax_idx].plot(iters[:len(cos_c_vals)], cos_c_vals, 'g-', label='cos_c (coarse)')
        if cos_f_vals:
            axes[ax_idx].plot(iters[:len(cos_f_vals)], cos_f_vals, 'm-', label='cos_f (fine)')
        axes[ax_idx].set_ylabel('Cosine Similarity')
        axes[ax_idx].set_title('Feature Similarity')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    # Radio cosine
    if radio_cos_vals:
        axes[ax_idx].plot(iters[:len(radio_cos_vals)], radio_cos_vals, 'orange', linewidth=1.5)
        axes[ax_idx].set_ylabel('RADIO Cosine')
        axes[ax_idx].set_title('RADIO Feature Alignment')
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    # Gaussian count
    if n_gaussians:
        axes[ax_idx].plot(iters[:len(n_gaussians)], n_gaussians, 'k-', linewidth=1.5)
        axes[ax_idx].set_ylabel('#Gaussians')
        axes[ax_idx].set_title('Gaussian Count')
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    axes[-1].set_xlabel('Iteration')
    fig.suptitle('Joint GSFF Training Log', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'joint_gsff_training.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Done] 训练曲线保存至: {save_path}')


def _parse_msflow_log(lines, output_dir):
    """解析 MSFlow 训练日志。"""
    epochs, rot_errs, trans_errs = [], [], []

    for line in lines:
        # [Val E10] med_rot=1.23° med_trans=5.67cm
        m = re.search(r'\[Val E(\d+)\].*?med_rot=([\d.]+).*?med_trans=([\d.]+)', line)
        if m:
            epochs.append(int(m.group(1)))
            rot_errs.append(float(m.group(2)))
            trans_errs.append(float(m.group(3)))

    if not epochs:
        print("[WARNING] 未解析到MSFlow验证数据")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(epochs, rot_errs, 'b-o', markersize=3)
    ax1.set_ylabel('Rotation Error (°)')
    ax1.set_title('Validation Rotation Error')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, trans_errs, 'r-o', markersize=3)
    ax2.set_ylabel('Translation Error (cm)')
    ax2.set_title('Validation Translation Error')
    ax2.set_xlabel('Epoch')
    ax2.grid(True, alpha=0.3)

    # Mark best
    best_idx = np.argmin(rot_errs)
    ax1.axvline(epochs[best_idx], color='green', linestyle='--', alpha=0.5)
    ax1.annotate(f'Best: {rot_errs[best_idx]:.2f}°', 
                 xy=(epochs[best_idx], rot_errs[best_idx]),
                 fontsize=10, color='green')

    fig.suptitle('MSFlow Training Log', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'msflow_training.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Done] 训练曲线保存至: {save_path}')


def _parse_2dgs_log(lines, output_dir):
    """解析 2DGS 训练日志。"""
    iters, psnrs = [], []

    for line in lines:
        m = re.search(r'(?:iter|step)[\s:]+(\d+).*?PSNR[\s:=]+([\d.]+)', line, re.IGNORECASE)
        if m:
            iters.append(int(m.group(1)))
            psnrs.append(float(m.group(2)))

    if not iters:
        print("[WARNING] 未解析到PSNR数据")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(iters, psnrs, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('2DGS Training PSNR')
    ax.grid(True, alpha=0.3)

    # Mark best
    best_idx = np.argmax(psnrs)
    ax.annotate(f'Best: {psnrs[best_idx]:.2f} dB @ iter {iters[best_idx]}',
                xy=(iters[best_idx], psnrs[best_idx]),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    save_path = os.path.join(output_dir, '2dgs_training.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Done] 训练曲线保存至: {save_path}')


def _parse_generic_log(lines, output_dir):
    """通用日志解析: 尝试提取所有数值字段。"""
    # 找所有 key=value 模式
    all_keys = set()
    for line in lines[:100]:
        matches = re.findall(r'(\w+)=([\d.]+)', line)
        for k, v in matches:
            try:
                float(v)
                all_keys.add(k)
            except ValueError:
                pass

    if not all_keys:
        print("[ERROR] 无法从日志中解析任何数值字段")
        return

    data = {k: [] for k in all_keys}
    line_indices = []

    for i, line in enumerate(lines):
        matches = dict(re.findall(r'(\w+)=([\d.]+)', line))
        if matches:
            line_indices.append(i)
            for k in all_keys:
                if k in matches:
                    data[k].append(float(matches[k]))
                elif data[k]:
                    data[k].append(data[k][-1])  # repeat last
                else:
                    data[k].append(0)

    # Plot top 6 varying keys
    variances = {k: np.var(v) if len(v) > 1 else 0 for k, v in data.items()}
    top_keys = sorted(variances, key=variances.get, reverse=True)[:6]

    n = len(top_keys)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, k in zip(axes, top_keys):
        ax.plot(range(len(data[k])), data[k], linewidth=1)
        ax.set_ylabel(k)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Log Line Index')
    fig.suptitle('Generic Log Parse', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'generic_log.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Done] 通用日志曲线保存至: {save_path}')


# ═══════════════════════════════════════════════════════════════════════════
# Mode 6: 特征质量分析
# ═══════════════════════════════════════════════════════════════════════════

def cmd_feature_quality(args):
    """
    特征质量分析: distinctiveness, PCA variance spectrum, channel统计。

    输入: 特征目录 (包含 .pt 文件)
    输出: 多面板分析图
    """
    from sklearn.decomposition import PCA

    output_dir = ensure_dir(args.output_dir)
    feature_dir = Path(args.feature_dir)
    feat_files = sorted(feature_dir.glob('*.pt'))

    n_samples = min(args.n_samples, len(feat_files))
    step = max(1, len(feat_files) // n_samples)
    selected = [feat_files[i * step] for i in range(n_samples)]

    print(f"分析 {n_samples} 个特征文件 from {feature_dir}")

    # 收集统计
    all_cos_sims = []  # 相邻帧余弦相似度
    all_channel_means = []
    all_channel_stds = []
    all_feats_for_pca = []

    for i, fpath in enumerate(selected):
        feat = load_feature(str(fpath))  # [C, H, W]
        C, H, W = feat.shape

        # Channel statistics
        all_channel_means.append(feat.mean(dim=(1, 2)).numpy())
        all_channel_stds.append(feat.std(dim=(1, 2)).numpy())

        # Collect for PCA spectrum
        flat = feat.reshape(C, -1).T  # [HW, C]
        # Sample pixels for PCA (avoid OOM for large features)
        if flat.shape[0] > 2000:
            idx = np.random.choice(flat.shape[0], 2000, replace=False)
            flat = flat[idx]
        all_feats_for_pca.append(flat.numpy())

        # Inter-frame cosine similarity
        if i > 0:
            prev_feat = load_feature(str(selected[i - 1]))
            if prev_feat.shape == feat.shape:
                sim = cosine_similarity_map(prev_feat, feat)
                all_cos_sims.append(sim.mean())

    # PCA variance spectrum
    all_feats = np.concatenate(all_feats_for_pca, axis=0)
    pca = PCA(n_components=min(all_feats.shape[1], 50))
    pca.fit(all_feats)

    # Generate multi-panel figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: PCA Variance Explained
    ax1 = fig.add_subplot(gs[0, 0])
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax1.bar(range(len(pca.explained_variance_ratio_)),
            pca.explained_variance_ratio_ * 100, alpha=0.7, label='Individual')
    ax1.plot(cumvar, 'r-', linewidth=2, label='Cumulative')
    ax1.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95%')
    ax1.set_xlabel('PC Index')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('PCA Variance Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Find 95% threshold
    n95 = np.searchsorted(cumvar, 95) + 1
    ax1.annotate(f'95% @ {n95} PCs', xy=(n95, 95), fontsize=10, color='green')

    # Panel 2: Channel Mean/Std
    ax2 = fig.add_subplot(gs[0, 1])
    means = np.array(all_channel_means)  # [N, C]
    stds = np.array(all_channel_stds)
    channel_ids = range(means.shape[1])
    ax2.errorbar(channel_ids, means.mean(0), yerr=means.std(0),
                 fmt='b-o', markersize=3, capsize=2, label='Mean ± std(mean)')
    ax2.errorbar(channel_ids, stds.mean(0), yerr=stds.std(0),
                 fmt='r-s', markersize=3, capsize=2, label='Std ± std(std)')
    ax2.set_xlabel('Channel Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Per-Channel Statistics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Inter-frame Cosine Similarity
    ax3 = fig.add_subplot(gs[1, 0])
    if all_cos_sims:
        ax3.plot(range(len(all_cos_sims)), all_cos_sims, 'g-o', markersize=4)
        ax3.set_xlabel('Frame Pair Index')
        ax3.set_ylabel('Mean Cosine Similarity')
        ax3.set_title(f'Inter-frame Similarity (avg={np.mean(all_cos_sims):.3f})')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'N/A (need >1 frame)', ha='center', va='center')

    # Panel 4: Distinctiveness (inverse of self-similarity)
    ax4 = fig.add_subplot(gs[1, 1])
    # Compute pixel distinctiveness for last loaded feature
    last_feat = load_feature(str(selected[-1]))
    C, H, W = last_feat.shape
    feat_norm = F.normalize(last_feat.float(), dim=0)  # [C, H, W]
    flat_norm = feat_norm.reshape(C, -1)  # [C, HW]
    # Self-similarity: mean cosine of each pixel to all others (sampled)
    n_query = min(200, H * W)
    query_idx = np.random.choice(H * W, n_query, replace=False)
    queries = flat_norm[:, query_idx]  # [C, n_query]
    sim_mat = (queries.T @ flat_norm).mean(dim=1)  # [n_query] avg similarity
    distinctiveness = 1.0 - sim_mat.numpy()

    ax4.hist(distinctiveness, bins=50, alpha=0.7, color='purple')
    ax4.set_xlabel('Distinctiveness (1 - avg_cosine)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Pixel Distinctiveness (mean={distinctiveness.mean():.3f})')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Feature Quality Analysis — {feature_dir.name}', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'feature_quality.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Text summary
    summary = {
        'feature_dir': str(feature_dir),
        'n_samples': n_samples,
        'feature_dim': int(all_feats.shape[1]),
        'pca_95pct_components': int(n95),
        'mean_inter_frame_cosine': float(np.mean(all_cos_sims)) if all_cos_sims else None,
        'mean_distinctiveness': float(distinctiveness.mean()),
    }
    with open(os.path.join(output_dir, 'feature_quality_report.txt'), 'w') as f:
        for k, v in summary.items():
            f.write(f'{k}: {v}\n')

    print(f'[Done] 特征质量分析保存至: {output_dir}')
    for k, v in summary.items():
        print(f'  {k}: {v}')


# ═══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='ICLPose 统一可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令说明:
  pca-features     PCA特征图可视化 (多尺度GT特征 → RGB)
  rgb-compare      GT vs Rendered RGB对比图 + Error map
  gsff-optim       GSFF位姿优化轨迹 (特征图 + loss曲线)
  cosine-map       特征余弦相似度热力图
  training-log     训练日志解析 → PSNR/loss曲线
  feature-quality  特征质量分析 (PCA spectrum + distinctiveness)

快速示例:
  python scripts/visualize_all.py pca-features --feature_dir output/features_multiscale/OldHospital
  python scripts/visualize_all.py training-log --log_path output/joint_gsff_radio/joint_oh_v8_gsff_radio/train.log
  python scripts/visualize_all.py feature-quality --feature_dir output/features_multiscale/OldHospital/coarse
        """
    )
    subparsers = parser.add_subparsers(dest='mode', help='可视化模式')

    # ── pca-features ──
    p_pca = subparsers.add_parser('pca-features', help='PCA特征图可视化')
    p_pca.add_argument('--feature_dir', required=True, help='特征根目录 (含 coarse/mid/fine_sd/fine_dino)')
    p_pca.add_argument('--output_dir', default='output/vis/pca_features')
    p_pca.add_argument('--frame_ids', nargs='+', type=int, help='指定帧ID (默认: 均匀采样)')
    p_pca.add_argument('--scales', nargs='+', help='要可视化的尺度 (默认: 全部)')
    p_pca.add_argument('--max_frames', type=int, default=10, help='最大帧数')

    # ── rgb-compare ──
    p_rgb = subparsers.add_parser('rgb-compare', help='GT vs Rendered RGB对比')
    p_rgb.add_argument('--ply_path', required=True, help='2DGS PLY文件路径')
    p_rgb.add_argument('--source_dir', required=True, help='场景数据目录')
    p_rgb.add_argument('--output_dir', default='output/vis/rgb_compare')
    p_rgb.add_argument('--n_views', type=int, default=8, help='渲染视图数')

    # ── gsff-optim ──
    p_gsff = subparsers.add_parser('gsff-optim', help='GSFF优化轨迹可视化')
    p_gsff.add_argument('--checkpoint', required=True, help='GSFF checkpoint路径')
    p_gsff.add_argument('--source_dir', required=True, help='场景数据目录')
    p_gsff.add_argument('--ply_path', required=True, help='2DGS PLY文件路径')
    p_gsff.add_argument('--cameras_json', required=True, help='cameras.json路径')
    p_gsff.add_argument('--output_dir', default='output/vis/gsff_optim')
    p_gsff.add_argument('--num_samples', type=int, default=4, help='样本数')

    # ── cosine-map ──
    p_cos = subparsers.add_parser('cosine-map', help='余弦相似度热力图')
    p_cos.add_argument('--feature_dir', required=True, help='特征目录A (含.pt文件)')
    p_cos.add_argument('--feature_dir_b', default=None, help='特征目录B (可选, 不指定则计算帧间自相似度)')
    p_cos.add_argument('--output_dir', default='output/vis/cosine_maps')
    p_cos.add_argument('--frame_ids', nargs='+', type=int, help='指定帧ID')

    # ── training-log ──
    p_log = subparsers.add_parser('training-log', help='训练日志曲线')
    p_log.add_argument('--log_path', required=True, help='训练日志路径')
    p_log.add_argument('--output_dir', default='output/vis/training_curves')

    # ── feature-quality ──
    p_qual = subparsers.add_parser('feature-quality', help='特征质量分析')
    p_qual.add_argument('--feature_dir', required=True, help='特征目录 (含.pt文件)')
    p_qual.add_argument('--output_dir', default='output/vis/feature_quality')
    p_qual.add_argument('--n_samples', type=int, default=50, help='采样帧数')

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        print("\n[ERROR] 请指定子命令, 例如: python scripts/visualize_all.py pca-features --help")
        return

    dispatch = {
        'pca-features': cmd_pca_features,
        'rgb-compare': cmd_rgb_compare,
        'gsff-optim': cmd_gsff_optim,
        'cosine-map': cmd_cosine_map,
        'training-log': cmd_training_log,
        'feature-quality': cmd_feature_quality,
    }

    print(f"{'=' * 60}")
    print(f"ICLPose 统一可视化工具 — Mode: {args.mode}")
    print(f"{'=' * 60}\n")

    dispatch[args.mode](args)


if __name__ == '__main__':
    main()
