"""
Feature Compression Quality Visualization
==========================================
对比原始融合特征 (768-dim) 与压缩/重建后特征 (256-dim→768-dim)，
验证 AutoEncoder 的压缩保真度。

输出 4 列并排图:
  [原始 PCA] | [重建 PCA] | [压缩特征 PCA] | [逐像素余弦相似度]

用法:
    python -m feature_compression.eval_compression \\
        --raw_dir   dataset/room_0/Sequence_1/features_raw \\
        --comp_dir  dataset/room_0/Sequence_1/features_compressed/fused \\
        --ae_path   dataset/room_0/Sequence_1/ae_models/ae_fused.pth \\
        --frame_ids 0 10 42 \\
        --output_dir output/compression_vis
"""

import sys
import re
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_compression.compressor import FeatureCompressor


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(raw_dir: str, frame_id: int) -> torch.Tensor:
    """加载原始 768-dim 融合特征，返回 [C, H, W]"""
    raw_dir = Path(raw_dir)
    pattern = f"rgb_{frame_id}_fused_*.pt"
    matches = list(raw_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Raw feature not found for frame {frame_id} in {raw_dir}")
    feat = torch.load(matches[0], map_location='cpu').float()
    if feat.dim() == 4:
        feat = feat.squeeze(0)   # [1, C, H, W] → [C, H, W]
    return feat                  # [768, 35, 46]


def load_compressed(comp_dir: str, frame_id: int) -> torch.Tensor:
    """加载压缩后 256-dim 特征，返回 [C, H, W]"""
    comp_dir = Path(comp_dir)
    pattern = f"rgb_{frame_id}_fused_*_compressed.pt"
    matches = list(comp_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Compressed feature not found for frame {frame_id} in {comp_dir}")
    data = torch.load(matches[0], map_location='cpu')
    if isinstance(data, dict):
        feat = data['compressed'].float()
    else:
        feat = data.float()
    if feat.dim() == 4:
        feat = feat.squeeze(0)
    # channels-last → channels-first
    if feat.dim() == 3 and feat.shape[-1] > feat.shape[0]:
        feat = feat.permute(2, 0, 1).contiguous()   # [H, W, 256] → [256, H, W]
    return feat   # [256, 35, 46]


# ─────────────────────────────────────────────────────────────────────────────
# PCA 工具
# ─────────────────────────────────────────────────────────────────────────────

def feat_to_pca_rgb(feat_a: np.ndarray, feat_b: np.ndarray,
                    extra_feats=None) -> list:
    """
    对多张特征图做联合 PCA → [0,1] RGB。
    feat_a, feat_b: [C, H, W]；extra_feats: list of [C, H, W]（可以是不同C，
    但都会单独做 PCA 到3维后返回）。

    返回: [pca_a, pca_b, ...] 每个均为 [H, W, 3] float in [0,1]
    """
    from sklearn.decomposition import PCA

    def _joint_pca(feats_list):
        """对同维度的特征列表做联合 PCA"""
        shapes = [f.shape for f in feats_list]
        D = shapes[0][0]
        pixels_list = [f.reshape(D, -1).T for f in feats_list]   # each [H*W, D]
        all_px = np.concatenate(pixels_list, axis=0)
        pca = PCA(n_components=3)
        pca.fit(all_px)
        results = []
        for f, (_, H, W) in zip(feats_list, shapes):
            proj = pca.transform(f.reshape(D, -1).T).reshape(H, W, 3)
            results.append(proj)
        # 联合归一化
        combined = np.concatenate([r.reshape(-1, 3) for r in results], axis=0)
        vmin = combined.min(0); vmax = combined.max(0)
        scale = np.where(vmax - vmin > 1e-8, vmax - vmin, 1.0)
        return [np.clip((r - vmin) / scale, 0, 1) for r in results]

    main_results = _joint_pca([feat_a, feat_b])
    extra_results = []
    if extra_feats:
        for ef in extra_feats:
            # 单独做 PCA（维度可能不同）
            D, H, W = ef.shape
            pca = PCA(n_components=3)
            proj = pca.fit_transform(ef.reshape(D, -1).T).reshape(H, W, 3)
            vmin = proj.min(); vmax_v = proj.max()
            proj = np.clip((proj - vmin) / max(vmax_v - vmin, 1e-8), 0, 1)
            extra_results.append(proj)
    return main_results + extra_results


# ─────────────────────────────────────────────────────────────────────────────
# 核心可视化
# ─────────────────────────────────────────────────────────────────────────────

def visualize_frame(
    frame_id: int,
    raw_dir: str,
    comp_dir: str,
    compressor: FeatureCompressor,
    output_dir: Path,
    rgb_dir: str = None,
):
    """对单帧做压缩质量可视化，保存对比图。"""

    # ── 1. 加载数据 ──
    feat_raw = load_raw(raw_dir, frame_id)          # [768, 35, 46]
    feat_comp = load_compressed(comp_dir, frame_id)  # [256, 35, 46]

    # ── 2. 解压重建 ──
    feat_recon = compressor.decompress(feat_comp)    # [768, 35, 46]

    # ── 3. 计算重建误差 ──
    # 余弦相似度（原始 vs 重建），归一化后逐像素计算
    raw_n   = F.normalize(feat_raw,   p=2, dim=0)
    recon_n = F.normalize(feat_recon, p=2, dim=0)
    cos_map = F.cosine_similarity(raw_n, recon_n, dim=0).numpy()   # [H, W]
    mean_cos = float(cos_map.mean())

    # L2 误差
    l2_map  = (feat_raw - feat_recon).norm(dim=0).numpy()          # [H, W]
    mean_l2 = float(l2_map.mean())

    # ── 4. PCA 可视化 ──
    raw_np   = feat_raw.numpy()    # [768, H, W]
    recon_np = feat_recon.numpy()  # [768, H, W]
    comp_np  = feat_comp.numpy()   # [256, H, W]

    try:
        pca_raw, pca_recon, pca_comp = feat_to_pca_rgb(
            raw_np, recon_np, extra_feats=[comp_np]
        )
    except ImportError:
        print("sklearn not installed. Run: pip install scikit-learn")
        return None

    # ── 5. 图像（可选背景）+ 确定显示分辨率 ──
    # 特征图来自把原图 resize 到 ceil(H/14)*14 x ceil(W/14)*14 后提取的 token grid。
    # 为了让特征图与 RGB 在空间上对应，将所有可视化图像上采样到原始分辨率。
    rgb_img = None
    display_h, display_w = None, None  # 将由 RGB 确定，或用特征图默认值
    if rgb_dir:
        rgb_path = Path(rgb_dir)
        candidates = sorted(rgb_path.glob(f"rgb_{frame_id}.*")) + \
                     sorted(rgb_path.glob(f"{frame_id:04d}.*")) + \
                     sorted(rgb_path.glob(f"frame{frame_id:06d}.*"))
        if candidates:
            from PIL import Image as PILImage
            pil_rgb = PILImage.open(candidates[0]).convert('RGB')
            # 以原始分辨率加载，不做 aspect-ratio 破坏性 resize
            display_w, display_h = pil_rgb.size   # PIL: (W, H)
            rgb_img = np.array(pil_rgb)

    # 将 PCA 图和余弦图上采样到显示分辨率（与 RGB 对齐）
    if display_h is not None:
        def _up(arr_hwc):
            """[H, W, C] float → 上采样到 display_h x display_w"""
            t = torch.from_numpy(arr_hwc).permute(2, 0, 1).unsqueeze(0).float()
            t = F.interpolate(t, size=(display_h, display_w),
                              mode='bilinear', align_corners=False)
            return t.squeeze(0).permute(1, 2, 0).numpy()
        def _up2d(arr_hw):
            """[H, W] float → 上采样到 display_h x display_w"""
            t = torch.from_numpy(arr_hw).unsqueeze(0).unsqueeze(0).float()
            t = F.interpolate(t, size=(display_h, display_w),
                              mode='bilinear', align_corners=False)
            return t.squeeze().numpy()
        pca_raw   = _up(pca_raw)
        pca_recon = _up(pca_recon)
        pca_comp  = _up(pca_comp)
        cos_map   = _up2d(cos_map)

    # ── 6. 绘图 ──
    n_cols = 5 if rgb_img is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    fig.suptitle(
        f"Feature Compression Quality  |  Frame {frame_id}  |"
        f"  Cosine Sim: {mean_cos:.4f}  |  L2 Error: {mean_l2:.4f}",
        fontsize=12, fontweight='bold'
    )

    col = 0
    if rgb_img is not None:
        axes[col].imshow(rgb_img)
        axes[col].set_title(f"RGB Image\n({display_w}×{display_h})", fontsize=10)
        axes[col].axis('off')
        col += 1

    axes[col].imshow(pca_raw)
    axes[col].set_title("Original (768-dim)\nPCA→RGB", fontsize=10)
    axes[col].axis('off')
    col += 1

    axes[col].imshow(pca_recon)
    axes[col].set_title(f"Reconstructed (768→256→768)\nPCA→RGB  [same PCA as orig]", fontsize=10)
    axes[col].axis('off')
    col += 1

    axes[col].imshow(pca_comp)
    axes[col].set_title("Compressed (256-dim)\nPCA→RGB  [independent PCA]", fontsize=10)
    axes[col].axis('off')
    col += 1

    im = axes[col].imshow(cos_map, cmap='RdYlGn', vmin=0.8, vmax=1.0)
    axes[col].set_title(f"Cosine Sim (orig vs recon)\nMean: {mean_cos:.4f}", fontsize=10)
    axes[col].axis('off')
    plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = output_dir / f"compression_frame_{frame_id:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Frame {frame_id}] cosine={mean_cos:.4f}  l2={mean_l2:.4f}  ->  {save_path}")
    return mean_cos


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Feature compression quality visualization")
    p.add_argument('--raw_dir',   type=str,
                   default='dataset/room_0/Sequence_1/features_raw',
                   help='Dir with rgb_{id}_fused_*.pt (original 768-dim)')
    p.add_argument('--comp_dir',  type=str,
                   default='dataset/room_0/Sequence_1/features_compressed/fused',
                   help='Dir with rgb_{id}_fused_*_compressed.pt (256-dim)')
    p.add_argument('--ae_path',   type=str,
                   default='dataset/room_0/Sequence_1/ae_models/ae_fused.pth',
                   help='AutoEncoder checkpoint path')
    p.add_argument('--output_dir', type=str,
                   default='output/compression_vis',
                   help='Output directory for visualization images')
    p.add_argument('--frame_ids', type=int, nargs='+', default=None,
                   help='Frame IDs to visualize (default: 0 10 42 100)')
    p.add_argument('--rgb_dir',   type=str, default=None,
                   help='Optional: RGB image dir for background reference')
    p.add_argument('--device',    type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = args.frame_ids or [0, 10, 42, 100]

    print(f"=== Feature Compression Visualization ===")
    print(f"  AE model : {args.ae_path}")
    print(f"  Frames   : {frame_ids}")

    device = args.device if torch.cuda.is_available() else 'cpu'
    compressor = FeatureCompressor(model_path=args.ae_path, device=device)

    scores = []
    for fid in frame_ids:
        try:
            s = visualize_frame(
                frame_id=fid,
                raw_dir=args.raw_dir,
                comp_dir=args.comp_dir,
                compressor=compressor,
                output_dir=output_dir,
                rgb_dir=args.rgb_dir,
            )
            if s is not None:
                scores.append(s)
        except FileNotFoundError as e:
            print(f"  [Frame {fid}] SKIP: {e}")

    if scores:
        print(f"\n{'─'*50}")
        print(f"  Mean cosine similarity: {np.mean(scores):.4f}")
        print(f"  Output dir: {output_dir.resolve()}")


if __name__ == '__main__':
    main()
