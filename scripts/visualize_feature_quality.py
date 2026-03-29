#!/usr/bin/env python3
"""
FlowFeat vs Baseline (SD+DINO) 特征质量可视化对比
生成 PCA 彩色图、余弦相似度热力图、渲染-查询对比、跨帧一致性图

输出: output/feature_vis/ 下多张对比图
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ── 路径配置 ──
FLOWFEAT_PCA_DIR = 'output/features_flowfeat_pca/OldHospital'
BASELINE_PER_SCALE = 'output/feature_3dgs/oldhospital_v5_stride7_fx1663/per_scale'
PLY_PATH = 'output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply'
TRIPLANE_PATH = 'output/feature_3dgs/oldhospital_triplane_v2/best_model.pth'
TRAJ_PATH = 'output/features_flowfeat_pca/OldHospital/traj_w_c.txt'

OUT_DIR = 'output/feature_vis'
os.makedirs(OUT_DIR, exist_ok=True)

# 帧索引
FRAME_INDICES = [0, 50, 100, 200, 500]


def load_poses(path):
    with open(path) as f:
        lines = f.readlines()
    poses = []
    for l in lines:
        vals = list(map(float, l.strip().split()))
        if len(vals) == 16:
            poses.append(np.array(vals).reshape(4, 4))
    return np.array(poses)


def load_flowfeat_query(frame_idx):
    """加载 FlowFeat PCA query 特征 (3 scales)."""
    feats = {}
    for scale, dim, h, w in [('coarse', 32, 20, 35), ('mid', 64, 40, 70), ('fine', 64, 80, 140)]:
        p = f'{FLOWFEAT_PCA_DIR}/{scale}/rgb_{frame_idx}_{scale}_{dim}x{h}x{w}.pt'
        feats[scale] = torch.load(p, map_location='cpu')  # (D, H, W)
    return feats


def pca_colorize(feat_map, n_components=3):
    """将 (D, H, W) 特征用 PCA 降到 3 维并映射到 RGB."""
    D, H, W = feat_map.shape
    X = feat_map.reshape(D, -1).T.numpy()  # (HW, D)
    pca = PCA(n_components=n_components)
    rgb = pca.fit_transform(X)  # (HW, 3)
    # 归一化到 [0, 1]
    for i in range(3):
        lo, hi = rgb[:, i].min(), rgb[:, i].max()
        if hi > lo:
            rgb[:, i] = (rgb[:, i] - lo) / (hi - lo)
    return rgb.reshape(H, W, 3)


def cosine_sim_map(feat_a, feat_b):
    """计算两个 (D, H, W) 特征图的逐像素余弦相似度."""
    # L2 normalize
    a = feat_a / (feat_a.norm(dim=0, keepdim=True) + 1e-8)
    b = feat_b / (feat_b.norm(dim=0, keepdim=True) + 1e-8)
    return (a * b).sum(dim=0).numpy()  # (H, W)


def render_flowfeat_triplane(pose_w2c_44, scales_res):
    """用 TriPlane 渲染 FlowFeat 特征."""
    from modules.multiscale_renderer import MultiScaleRenderer
    renderer = MultiScaleRenderer(
        ply_path=PLY_PATH,
        scale_model_paths={},
        device=DEVICE,
        img_height=1080, img_width=1920,
        fx=1663.12, fy=1663.12, cx=960.0, cy=540.0,
        triplane_model_path=TRIPLANE_PATH,
        scale_resolutions=scales_res,
    )
    result = renderer.render_all_scales(pose_w2c_44, scales=list(scales_res.keys()), return_depth=False)
    feats = {}
    for s in scales_res:
        feats[s] = result[f'{s}_feat'].cpu()  # (D, H, W)
    depth = None
    return feats, depth


def render_baseline_perscale(pose_w2c_44):
    """用 per-Gaussian 渲染 baseline SD+DINO 特征."""
    from modules.multiscale_renderer import MultiScaleRenderer
    scale_paths = {
        'coarse': f'{BASELINE_PER_SCALE}/coarse.pth',
        'mid': f'{BASELINE_PER_SCALE}/mid.pth',
        'fine_sd': f'{BASELINE_PER_SCALE}/fine_sd.pth',
        'fine_dino': f'{BASELINE_PER_SCALE}/fine_dino.pth',
    }
    renderer = MultiScaleRenderer(
        ply_path=PLY_PATH,
        scale_model_paths=scale_paths,
        device=DEVICE,
        img_height=1080, img_width=1920,
        fx=1663.12, fy=1663.12, cx=960.0, cy=540.0,
    )
    result = renderer.render_all_scales(pose_w2c_44, scales=list(scale_paths.keys()), return_depth=False)
    feats = {}
    for s in scale_paths:
        feats[s] = result[f'{s}_feat'].cpu()
    return feats


def load_baseline_query_approx(frame_idx):
    """加载 baseline per-Gaussian 的 query 特征 (使用 v1 格式如果存在)."""
    # 尝试查找 original multiscale features
    v1_dir = 'output/features_multiscale/OldHospital'
    if not os.path.isdir(v1_dir):
        return None
    feats = {}
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        scale_dir = os.path.join(v1_dir, scale)
        if os.path.isdir(scale_dir):
            files = sorted(os.listdir(scale_dir))
            if frame_idx < len(files):
                feats[scale] = torch.load(os.path.join(scale_dir, files[frame_idx]), map_location='cpu')
    return feats if feats else None


# ═══════════════════════════════════════════════════════
# 可视化 1: FlowFeat PCA 彩色图 (query 特征, 多帧多尺度)
# ═══════════════════════════════════════════════════════
def vis1_query_pca_colormap():
    print("=== VIS1: FlowFeat Query PCA Colormap ===")
    fig, axes = plt.subplots(len(FRAME_INDICES), 3, figsize=(15, 3 * len(FRAME_INDICES)))
    scales = ['coarse', 'mid', 'fine']

    for row, fi in enumerate(FRAME_INDICES):
        feats = load_flowfeat_query(fi)
        for col, scale in enumerate(scales):
            rgb = pca_colorize(feats[scale])
            axes[row, col].imshow(rgb)
            axes[row, col].set_title(f'Frame {fi} / {scale} ({feats[scale].shape[0]}d)', fontsize=9)
            axes[row, col].axis('off')

    fig.suptitle('FlowFeat PCA Query Features (PCA→RGB)', fontsize=14)
    fig.tight_layout()
    path = f'{OUT_DIR}/vis1_flowfeat_query_pca.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════
# 可视化 2: 渲染 vs 查询 余弦相似度热力图
# ═══════════════════════════════════════════════════════
def vis2_render_vs_query_cosine():
    print("=== VIS2: Rendered vs Query Cosine Similarity ===")
    poses_c2w = load_poses(TRAJ_PATH)
    scales_res = {'coarse': [20, 35], 'mid': [40, 70], 'fine': [80, 140]}
    scales = ['coarse', 'mid', 'fine']

    frames = FRAME_INDICES[:3]  # 只渲染 3 帧减少时间
    fig, axes = plt.subplots(len(frames), 3, figsize=(15, 4 * len(frames)))

    for row, fi in enumerate(frames):
        T_c2w = poses_c2w[fi]
        T_w2c = np.linalg.inv(T_c2w)
        T_w2c_t = torch.from_numpy(T_w2c).float().to(DEVICE)

        query = load_flowfeat_query(fi)
        rendered, depth = render_flowfeat_triplane(T_w2c_t, scales_res)

        for col, scale in enumerate(scales):
            sim = cosine_sim_map(query[scale], rendered[scale])
            im = axes[row, col].imshow(sim, vmin=0.3, vmax=1.0, cmap='RdYlGn')
            axes[row, col].set_title(f'Frame {fi} / {scale}\nmean={sim.mean():.3f}', fontsize=9)
            axes[row, col].axis('off')
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    fig.suptitle('FlowFeat: Rendered vs Query Cosine Similarity (TriPlane)', fontsize=14)
    fig.tight_layout()
    path = f'{OUT_DIR}/vis2_render_query_cosine.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════
# 可视化 3: 渲染 vs 查询 PCA 彩色对比 (同帧)
# ═══════════════════════════════════════════════════════
def vis3_render_vs_query_pca():
    print("=== VIS3: Rendered vs Query PCA Comparison ===")
    poses_c2w = load_poses(TRAJ_PATH)
    scales_res = {'coarse': [20, 35], 'mid': [40, 70], 'fine': [80, 140]}
    scales = ['coarse', 'mid', 'fine']

    fi = FRAME_INDICES[0]
    T_c2w = poses_c2w[fi]
    T_w2c = np.linalg.inv(T_c2w)
    T_w2c_t = torch.from_numpy(T_w2c).float().to(DEVICE)

    query = load_flowfeat_query(fi)
    rendered, _ = render_flowfeat_triplane(T_w2c_t, scales_res)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for col, scale in enumerate(scales):
        # 使用同一个 PCA 让颜色可比
        D, H, W = query[scale].shape
        q_flat = query[scale].reshape(D, -1).T.numpy()
        r_flat = rendered[scale].reshape(D, -1).T.numpy()

        combined = np.concatenate([q_flat, r_flat], axis=0)
        pca = PCA(n_components=3)
        pca.fit(combined)

        q_rgb = pca.transform(q_flat)
        r_rgb = pca.transform(r_flat)
        for arr in [q_rgb, r_rgb]:
            for i in range(3):
                lo, hi = combined[:, 0].min(), combined[:, 0].max()  # global range
                lo_i, hi_i = arr[:, i].min(), arr[:, i].max()
                if hi_i > lo_i:
                    arr[:, i] = (arr[:, i] - lo_i) / (hi_i - lo_i)

        axes[0, col].imshow(q_rgb.reshape(H, W, 3))
        axes[0, col].set_title(f'Query / {scale}', fontsize=10)
        axes[0, col].axis('off')

        rH, rW = rendered[scale].shape[1], rendered[scale].shape[2]
        axes[1, col].imshow(r_rgb.reshape(rH, rW, 3))
        axes[1, col].set_title(f'Rendered / {scale}', fontsize=10)
        axes[1, col].axis('off')

    fig.suptitle(f'Frame {fi}: Query vs TriPlane Rendered (shared PCA→RGB)', fontsize=14)
    fig.tight_layout()
    path = f'{OUT_DIR}/vis3_render_query_pca_compare.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════
# 可视化 4: 跨帧一致性 (query 特征间余弦)
# ═══════════════════════════════════════════════════════
def vis4_cross_frame_consistency():
    print("=== VIS4: Cross-Frame Query Feature Consistency ===")
    scales = ['coarse', 'mid', 'fine']
    n_frames = 10
    frame_ids = list(range(0, n_frames * 5, 5))  # 0, 5, 10, ..., 45

    # Load all query features
    all_feats = {}
    for scale in scales:
        feats_list = []
        for fi in frame_ids:
            f = load_flowfeat_query(fi)
            feats_list.append(f[scale])
        all_feats[scale] = feats_list

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for col, scale in enumerate(scales):
        n = len(frame_ids)
        cos_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a = all_feats[scale][i]
                b = all_feats[scale][j]
                a_norm = a / (a.norm(dim=0, keepdim=True) + 1e-8)
                b_norm = b / (b.norm(dim=0, keepdim=True) + 1e-8)
                cos_matrix[i, j] = (a_norm * b_norm).sum(dim=0).mean().item()

        im = axes[col].imshow(cos_matrix, vmin=0.3, vmax=1.0, cmap='RdYlGn')
        axes[col].set_title(f'{scale}\ndiag_mean={np.diag(cos_matrix).mean():.3f}', fontsize=10)
        axes[col].set_xlabel('Frame idx')
        axes[col].set_ylabel('Frame idx')
        axes[col].set_xticks(range(n))
        axes[col].set_xticklabels(frame_ids, fontsize=7, rotation=45)
        axes[col].set_yticks(range(n))
        axes[col].set_yticklabels(frame_ids, fontsize=7)
        fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

    fig.suptitle('FlowFeat: Cross-Frame Query Cosine Similarity Matrix', fontsize=14)
    fig.tight_layout()
    path = f'{OUT_DIR}/vis4_crossframe_consistency.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════
# 可视化 5: Baseline 渲染 vs FlowFeat 渲染 PCA 对比
# ═══════════════════════════════════════════════════════
def vis5_baseline_vs_flowfeat_rendered():
    print("=== VIS5: Baseline Rendered vs FlowFeat Rendered ===")
    poses_c2w = load_poses(TRAJ_PATH)
    fi = FRAME_INDICES[0]
    T_c2w = poses_c2w[fi]
    T_w2c = np.linalg.inv(T_c2w)
    T_w2c_t = torch.from_numpy(T_w2c).float().to(DEVICE)

    # FlowFeat rendered (triplane)
    scales_res = {'coarse': [20, 35], 'mid': [40, 70], 'fine': [80, 140]}
    ff_rendered, ff_depth = render_flowfeat_triplane(T_w2c_t, scales_res)

    # Baseline rendered (per-Gaussian)
    bl_rendered = render_baseline_perscale(T_w2c_t)

    # 2 rows: FlowFeat (coarse/mid/fine) + Baseline (coarse/mid/fine_sd+fine_dino)
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))

    ff_scales = ['coarse', 'mid', 'fine']
    bl_scales = ['coarse', 'mid', 'fine_sd', 'fine_dino']

    for col, scale in enumerate(ff_scales):
        rgb = pca_colorize(ff_rendered[scale])
        axes[0, col].imshow(rgb)
        d = ff_rendered[scale].shape[0]
        h, w = ff_rendered[scale].shape[1], ff_rendered[scale].shape[2]
        axes[0, col].set_title(f'FlowFeat / {scale}\n{d}d @ {h}×{w}', fontsize=9)
        axes[0, col].axis('off')
    axes[0, 3].axis('off')
    axes[0, 3].text(0.5, 0.5, 'N/A\n(3 scales)', ha='center', va='center', fontsize=12, color='gray')

    for col, scale in enumerate(bl_scales):
        rgb = pca_colorize(bl_rendered[scale])
        axes[1, col].imshow(rgb)
        d = bl_rendered[scale].shape[0]
        h, w = bl_rendered[scale].shape[1], bl_rendered[scale].shape[2]
        axes[1, col].set_title(f'Baseline / {scale}\n{d}d @ {h}×{w}', fontsize=9)
        axes[1, col].axis('off')

    fig.suptitle(f'Frame {fi}: FlowFeat TriPlane vs Baseline Per-Gaussian Rendering', fontsize=14)
    fig.tight_layout()
    path = f'{OUT_DIR}/vis5_flowfeat_vs_baseline_rendered.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════
# 可视化 6: 分辨率对比条形图 + 数值统计表
# ═══════════════════════════════════════════════════════
def vis6_stats_comparison():
    print("=== VIS6: Statistics Comparison ===")
    poses_c2w = load_poses(TRAJ_PATH)
    scales_res = {'coarse': [20, 35], 'mid': [40, 70], 'fine': [80, 140]}

    # 采样多帧计算统计
    sample_frames = [0, 50, 100, 200, 500]
    ff_cos = {'coarse': [], 'mid': [], 'fine': []}
    bl_cos = {'coarse': [], 'mid': [], 'fine_sd': [], 'fine_dino': []}

    for fi in sample_frames:
        T_c2w = poses_c2w[fi]
        T_w2c = np.linalg.inv(T_c2w)
        T_w2c_t = torch.from_numpy(T_w2c).float().to(DEVICE)

        # FlowFeat
        q_ff = load_flowfeat_query(fi)
        r_ff, _ = render_flowfeat_triplane(T_w2c_t, scales_res)
        for s in ['coarse', 'mid', 'fine']:
            sim = cosine_sim_map(q_ff[s], r_ff[s])
            ff_cos[s].append(sim.mean())

        # Baseline: render + approximate query
        r_bl = render_baseline_perscale(T_w2c_t)
        q_bl = load_baseline_query_approx(fi)
        if q_bl:
            for s in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
                if s in q_bl and s in r_bl:
                    # Match spatial size
                    q_feat = q_bl[s]
                    r_feat = r_bl[s]
                    if q_feat.shape != r_feat.shape:
                        import torch.nn.functional as F
                        q_feat = F.interpolate(
                            q_feat.unsqueeze(0), size=r_feat.shape[1:],
                            mode='bilinear', align_corners=False).squeeze(0)
                    sim = cosine_sim_map(q_feat, r_feat)
                    bl_cos[s].append(sim.mean())

    # 绘制条形图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # FlowFeat cosine bars
    ff_means = {s: np.mean(v) for s, v in ff_cos.items()}
    ff_stds = {s: np.std(v) for s, v in ff_cos.items()}
    x_ff = list(ff_means.keys())
    y_ff = [ff_means[s] for s in x_ff]
    e_ff = [ff_stds[s] for s in x_ff]
    bars1 = ax1.bar(x_ff, y_ff, yerr=e_ff, color=['#e74c3c', '#e67e22', '#f39c12'], capsize=5)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Mean Cosine Similarity')
    ax1.set_title('FlowFeat (TriPlane)\nRendered vs Query')
    for bar, v in zip(bars1, y_ff):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Baseline cosine bars
    bl_means = {s: np.mean(v) for s, v in bl_cos.items() if v}
    if bl_means:
        bl_stds = {s: np.std(v) for s, v in bl_cos.items() if v}
        x_bl = list(bl_means.keys())
        y_bl = [bl_means[s] for s in x_bl]
        e_bl = [bl_stds[s] for s in x_bl]
        bars2 = ax2.bar(x_bl, y_bl, yerr=e_bl, color=['#2ecc71', '#27ae60', '#1abc9c', '#16a085'], capsize=5)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel('Mean Cosine Similarity')
        ax2.set_title('Baseline (SD+DINO, Per-Gaussian)\nRendered vs Query')
        for bar, v in zip(bars2, y_bl):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Baseline query features\nnot available', ha='center', va='center',
                fontsize=14, color='gray', transform=ax2.transAxes)
        ax2.set_title('Baseline (SD+DINO)')

    fig.suptitle('Feature Quality: Rendered vs Query Cosine (Same Pose)', fontsize=14)
    fig.tight_layout()
    path = f'{OUT_DIR}/vis6_cosine_comparison_bars.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # 打印数值表
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │  FlowFeat (Rendered vs Query) Mean Cosine   │")
    print("  ├─────────┬──────────┬─────────┬──────────────┤")
    print(f"  │ coarse  │ {ff_means.get('coarse',0):.4f}   │ mid     │ {ff_means.get('mid',0):.4f}       │")
    print(f"  │ fine    │ {ff_means.get('fine',0):.4f}   │         │              │")
    print("  ├─────────────────────────────────────────────┤")
    if bl_means:
        print("  │  Baseline (Rendered vs Query) Mean Cosine   │")
        print("  ├─────────┬──────────┬─────────┬──────────────┤")
        for s in bl_means:
            print(f"  │ {s:9s}│ {bl_means[s]:.4f}   │         │              │")
    print("  └─────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════
# 可视化 7: TriPlane 渲染深度图
# ═══════════════════════════════════════════════════════
def vis7_depth_maps():
    print("=== VIS7: Rendered Depth Maps ===")
    poses_c2w = load_poses(TRAJ_PATH)
    scales_res = {'coarse': [20, 35], 'mid': [40, 70], 'fine': [80, 140]}

    frames = FRAME_INDICES[:4]
    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))

    for col, fi in enumerate(frames):
        T_c2w = poses_c2w[fi]
        T_w2c = np.linalg.inv(T_c2w)
        T_w2c_t = torch.from_numpy(T_w2c).float().to(DEVICE)

        _, depth = render_flowfeat_triplane(T_w2c_t, scales_res)
        if depth is not None:
            d_np = depth.numpy()
            valid = d_np > 0
            im = axes[col].imshow(d_np, cmap='viridis')
            axes[col].set_title(f'Frame {fi}\n'
                               f'range=[{d_np[valid].min():.1f}, {d_np[valid].max():.1f}]m',
                               fontsize=9)
            fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
        else:
            axes[col].text(0.5, 0.5, 'No depth', ha='center', va='center')
        axes[col].axis('off')

    fig.suptitle('TriPlane Rendered Depth Maps', fontsize=14)
    fig.tight_layout()
    path = f'{OUT_DIR}/vis7_depth_maps.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    print("=" * 60)
    print("  FlowFeat vs Baseline Feature Quality Visualization")
    print("=" * 60)

    # Query PCA 彩色图 (不需要 GPU 渲染)
    vis1_query_pca_colormap()

    # 跨帧一致性矩阵 (不需要 GPU 渲染)
    vis4_cross_frame_consistency()

    # 需要 GPU 渲染的可视化
    vis2_render_vs_query_cosine()
    vis3_render_vs_query_pca()
    vis5_baseline_vs_flowfeat_rendered()
    vis6_stats_comparison()
    # vis7 skipped: rasterization_2dgs unavailable in gsplat 1.2.0

    print(f"\n✅ All visualizations saved to {OUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.png'):
            print(f"  {f}")
