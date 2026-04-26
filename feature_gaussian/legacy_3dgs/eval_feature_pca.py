"""
Feature 3DGS PCA Visualization
================================
对训练好的 Feature3DGS 进行验证：
  1. 渲染指定视角的特征图 (rendered_feat)
  2. 加载对应帧的真值特征图 (gt_feat)
  3. 对二者做联合 PCA 降维至 3 维 → 可视化为 RGB 图像
  4. 并排保存，方便直观比较

用法:
    python -m feature_3dgs.eval_feature_pca \\
        --ply_path output/feature_3dgs/room_0/point_cloud_with_features.ply \\
        --feature_dir dataset/room_0/Sequence_1/features_compressed/fused \\
        --traj_path  dataset/room_0/Sequence_1/traj_w_c.txt \\
        --frame_id   42 \\
        --output_dir output/feature_3dgs/pca_vis

高级选项:
    --frames  0 10 50 100   # 可视化多帧
    --feature_dim 256
    --no_show               # 不弹窗，只保存图片
"""

import os
import sys
import re
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # 无头服务器默认 Agg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 确保能 import 项目模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_gaussian.legacy_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_gaussian.legacy_3dgs.feature_renderer import FeatureRenderer


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_traj(traj_path: str) -> np.ndarray:
    """加载位姿文件，返回 W2C [N, 4, 4]。traj_w_c.txt 存储的是 C2W。"""
    traj = np.loadtxt(traj_path)
    c2w = traj.reshape(-1, 4, 4).astype(np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)   # [N, 4, 4]
    return w2c


def find_feature_file(feature_dir: str, frame_id: int):
    """在 feature_dir 中查找指定帧的特征文件。"""
    feature_dir = Path(feature_dir)
    # 匹配模式: rgb_{frame_id}_fused_*_compressed.pt
    pattern = f"rgb_{frame_id}_fused_*_compressed.pt"
    matches = list(feature_dir.glob(pattern))
    if not matches:
        # 也搜索无 compressed 后缀的
        pattern2 = f"rgb_{frame_id}_fused_*.pt"
        matches = list(feature_dir.glob(pattern2))
    if not matches:
        raise FileNotFoundError(
            f"找不到帧 {frame_id} 的特征文件，目录: {feature_dir}"
        )
    return matches[0]


def load_gt_feature(feature_path: str, normalize: bool = True) -> torch.Tensor:
    """加载真值特征图，返回 [D, H, W] float32 tensor。

    与 FeatureEmbeddingDataset.__getitem__ 保持完全一致的加载逻辑：
      - dict 格式: 优先取 'compressed' 键 (训练脚本保存格式)
      - layout 检测: 若 shape[-1] == 256 → channels-last [H, W, D] → permute [D, H, W]
    """
    data = torch.load(feature_path, map_location='cpu')

    if isinstance(data, dict):
        # 优先级: compressed > feature > feat > 第一个值
        feat = data.get('compressed',
               data.get('feature',
               data.get('feat', next(iter(data.values())))))
    else:
        feat = data
    feat = feat.float()

    # 去掉 batch 维
    if feat.dim() == 4:
        feat = feat.squeeze(0)

    # 与 FeatureEmbeddingDataset 保持一致: 若 shape[-1] >> shape[0] → channels-last
    # 典型尺寸: [35, 46, 256] → 256 > 35 触发转置
    if feat.dim() == 3 and feat.shape[-1] > feat.shape[0]:
        feat = feat.permute(2, 0, 1).contiguous()

    if normalize:
        feat = F.normalize(feat, p=2, dim=0)
    return feat  # [D, H, W]


def joint_pca(feat_a: np.ndarray, feat_b: np.ndarray, n_components: int = 3):
    """
    对两张特征图做联合 PCA 降至 n_components 维。

    Args:
        feat_a: [D, H1, W1] rendered feature map (numpy)
        feat_b: [D, H2, W2] GT feature map (numpy) 可与 a 尺寸不同
        n_components: 输出维度

    Returns:
        pca_a: [H1, W1, n_components] 归一化至 [0,1]
        pca_b: [H2, W2, n_components] 归一化至 [0,1]
    """
    from sklearn.decomposition import PCA

    D, H1, W1 = feat_a.shape
    D2, H2, W2 = feat_b.shape
    assert D == D2, f"特征维度不匹配: {D} vs {D2}"

    # 展平像素并合并
    pixels_a = feat_a.reshape(D, -1).T   # [H1*W1, D]
    pixels_b = feat_b.reshape(D, -1).T   # [H2*W2, D]
    all_pixels = np.concatenate([pixels_a, pixels_b], axis=0)  # [N, D]

    pca = PCA(n_components=n_components)
    pca.fit(all_pixels)

    proj_a = pca.transform(pixels_a).reshape(H1, W1, n_components)
    proj_b = pca.transform(pixels_b).reshape(H2, W2, n_components)

    # 联合归一化至 [0, 1]
    combined = np.concatenate([proj_a.reshape(-1, n_components),
                                proj_b.reshape(-1, n_components)], axis=0)
    vmin = combined.min(axis=0)
    vmax = combined.max(axis=0)
    scale = np.where(vmax - vmin > 1e-8, vmax - vmin, 1.0)

    pca_a = np.clip((proj_a - vmin) / scale, 0, 1)
    pca_b = np.clip((proj_b - vmin) / scale, 0, 1)

    return pca_a, pca_b


def cosine_similarity_map(feat_rendered: np.ndarray, feat_gt: np.ndarray) -> np.ndarray:
    """
    计算逐像素余弦相似度。feat_rendered 会先 resize 到 feat_gt 尺寸。
    输入: [D, H, W]，输出: [H, W] in [-1, 1]。
    """
    import torch
    import torch.nn.functional as tnF

    t_r = torch.from_numpy(feat_rendered).unsqueeze(0)  # [1, D, H1, W1]
    t_g = torch.from_numpy(feat_gt)                      # [D, H2, W2]
    H2, W2 = t_g.shape[1], t_g.shape[2]

    # resize rendered to gt resolution
    t_r_resized = tnF.interpolate(t_r, size=(H2, W2), mode='bilinear',
                                   align_corners=False).squeeze(0)  # [D, H2, W2]

    cos = tnF.cosine_similarity(t_r_resized, t_g, dim=0)  # [H2, W2]
    return cos.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 核心: 渲染 + 对比可视化
# ─────────────────────────────────────────────────────────────────────────────

def visualize_frame(
    model: GaussianFeatureModel,
    w2c: np.ndarray,                  # [4, 4]
    gt_feat: torch.Tensor,            # [D, fH, fW]
    frame_id: int,
    output_dir: Path,
    fx: float, fy: float, cx: float, cy: float,
    img_height: int = 480, img_width: int = 640,
    show: bool = False,
):
    """渲染一帧特征图，与真值对比可视化。"""
    device = next(iter([model.get_xyz])).device

    viewmat = torch.from_numpy(w2c).float().to(device)  # [4, 4]

    # ---- 上采样 GT 特征到全分辨率（与训练保持一致）----
    gt_feat_full = F.interpolate(
        gt_feat.unsqueeze(0),
        size=(img_height, img_width),
        mode='bilinear', align_corners=False,
    ).squeeze(0)
    gt_feat_full = F.normalize(gt_feat_full, p=2, dim=0)  # [D, H, W]

    # ---- 渲染特征图（全分辨率，与训练保持一致）----
    with torch.no_grad():
        result = FeatureRenderer.render_features(
            gaussian_model=model,
            viewmat=viewmat,
            fx=fx, fy=fy, cx=cx, cy=cy,
            img_height=img_height,
            img_width=img_width,
            feature_height=img_height,
            feature_width=img_width,
            norm_feat_before_render=True,
            norm_feat_after_render=True,
        )
    feat_rendered = result['feature_map']   # [D, H, W]

    feat_r_np = feat_rendered.cpu().numpy()   # [D, H, W]
    gt_np = gt_feat_full.numpy()              # [D, H, W]

    # ---- 联合 PCA 可视化 ----
    try:
        pca_r, pca_g = joint_pca(feat_r_np, gt_np, n_components=3)
    except ImportError:
        print("⚠️  sklearn not installed. Run: pip install scikit-learn")
        return

    # ---- 余弦相似度（两者分辨率完全一致，无需 resize）----
    cos_sim = F.cosine_similarity(feat_rendered.cpu(), gt_feat_full, dim=0).numpy()  # [H, W]
    mean_cos = float(cos_sim.mean())

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Feature 3DGS Validation  |  Frame {frame_id}  |  Mean Cosine Sim: {mean_cos:.4f}",
        fontsize=13, fontweight='bold'
    )

    axes[0].imshow(pca_r)
    axes[0].set_title("Rendered Feature (PCA->RGB)", fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(pca_g)
    axes[1].set_title("GT Feature Map (PCA->RGB)", fontsize=11)
    axes[1].axis('off')

    im = axes[2].imshow(cos_sim, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[2].set_title(f"Per-pixel Cosine Similarity\nMean: {mean_cos:.4f}", fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    save_path = output_dir / f"pca_frame_{frame_id:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [帧 {frame_id}] 余弦相似度: {mean_cos:.4f}  →  {save_path}")

    if show:
        matplotlib.use('TkAgg')
        plt.show()
    plt.close(fig)

    return mean_cos


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature3DGS PCA 可视化验证工具"
    )
    parser.add_argument('--ply_path', type=str, required=True,
                        help='带特征的 PLY 文件路径 (含 loc_* 属性)')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='真值压缩特征目录 (含 rgb_*_fused_*_compressed.pt)')
    parser.add_argument('--traj_path', type=str, required=True,
                        help='相机位姿文件 traj_w_c.txt (C2W 格式)')
    parser.add_argument('--output_dir', type=str,
                        default='output/feature_3dgs/pca_vis',
                        help='可视化结果输出目录')
    parser.add_argument('--frame_id', type=int, default=None,
                        help='指定单帧 ID (与 --frames 互斥)')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='指定多帧 ID 列表，如: --frames 0 10 50 100')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='特征维度 (default: 256)')
    # 相机参数 (Replica room_0 默认值)
    parser.add_argument('--fx', type=float, default=320.0)
    parser.add_argument('--fy', type=float, default=320.0)
    parser.add_argument('--cx', type=float, default=319.5)
    parser.add_argument('--cy', type=float, default=239.5)
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width',  type=int, default=640)
    parser.add_argument('--no_show', action='store_true',
                        help='不弹窗，仅保存图片')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 确定要可视化的帧 ──
    if args.frames is not None:
        frame_ids = args.frames
    elif args.frame_id is not None:
        frame_ids = [args.frame_id]
    else:
        # 默认可视化第 0、10、50、100 帧 (如存在)
        frame_ids = [0, 10, 50, 100]
        print(f"未指定帧 ID，将尝试: {frame_ids}")

    # ── 加载模型 ──
    print(f"\n=== 加载 Feature3DGS 模型 ===")
    print(f"  PLY: {args.ply_path}")
    model = GaussianFeatureModel(feature_dim=args.feature_dim)
    model.load_ply_with_features(args.ply_path)
    model = model.to(device)
    model.eval()
    print(f"  Gaussians: {model.get_xyz.shape[0]:,}  |  特征维度: {model.feature_dim}")

    # ── 加载位姿 ──
    print(f"\n=== 加载位姿 ===")
    w2c_poses = load_traj(args.traj_path)   # [N, 4, 4]
    print(f"  共 {len(w2c_poses)} 帧位姿")

    # ── 逐帧可视化 ──
    print(f"\n=== 开始 PCA 可视化 ===")
    cos_scores = []

    for frame_id in frame_ids:
        if frame_id >= len(w2c_poses):
            print(f"  ⚠  帧 {frame_id} 超出位姿范围 ({len(w2c_poses)} 帧)，跳过")
            continue

        try:
            feat_file = find_feature_file(args.feature_dir, frame_id)
        except FileNotFoundError as e:
            print(f"  ⚠  {e}，跳过")
            continue

        gt_feat = load_gt_feature(str(feat_file), normalize=True)  # [D, fH, fW]
        w2c = w2c_poses[frame_id]

        score = visualize_frame(
            model=model,
            w2c=w2c,
            gt_feat=gt_feat,
            frame_id=frame_id,
            output_dir=output_dir,
            fx=args.fx, fy=args.fy,
            cx=args.cx, cy=args.cy,
            img_height=args.img_height,
            img_width=args.img_width,
            show=not args.no_show,
        )
        if score is not None:
            cos_scores.append(score)

    if cos_scores:
        print(f"\n{'─'*50}")
        print(f"  Mean cosine similarity ({len(cos_scores)} frames): {np.mean(cos_scores):.4f}")
        print(f"  Max / Min: {max(cos_scores):.4f} / {min(cos_scores):.4f}")
        print(f"  Output dir: {output_dir.resolve()}")
    else:
        print("  No frames processed successfully")


if __name__ == '__main__':
    main()
