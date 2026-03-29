"""
DA3 特征渲染对比图生成
======================
生成 GT vs Rendered 特征的 PCA RGB 可视化对比图。
每个尺度(coarse/mid/fine)、每个版本的渲染结果与GT并排显示。

Output: output/da3_feature_vis/<version>_<scale>_comparison.png
"""

import sys, os, re
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.decomposition import PCA

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer

# ============================================================
# 配置
# ============================================================
PLY_PATH = "output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply"
FEATURE_DIR = "output/features_da3/OldHospital_indexed"
TRAJ_PATH = "output/features_da3/OldHospital_indexed/traj_w_c.txt"
IMG_W, IMG_H = 1920, 1080
FX, FY, CX, CY = 1663.12, 1663.12, 960.0, 540.0

SCALE_DIMS = {'coarse': 32, 'mid': 64, 'fine': 64}

MODEL_VERSIONS = {
    'v1': 'output/feature_3dgs/oldhospital_da3_perscale',
    'v2': 'output/feature_3dgs/oldhospital_da3_perscale_v2',
    'v3': 'output/feature_3dgs/oldhospital_da3_perscale_v3',
}

TEST_FRAMES = [0, 100, 300, 500, 800, 1000]
OUTPUT_DIR = "output/da3_feature_vis"


def pca_colorize(feat_map, pca_model=None, return_pca=False):
    """将 [D, H, W] 特征图 PCA 降到 3 通道，归一化到 [0,1] 的 RGB。"""
    D, H, W = feat_map.shape
    flat = feat_map.reshape(D, -1).T.cpu().numpy()  # [HW, D]
    
    if pca_model is None:
        pca_model = PCA(n_components=3)
        pca_model.fit(flat)
    
    rgb = pca_model.transform(flat)  # [HW, 3]
    # 归一化到 [0,1]
    for c in range(3):
        lo, hi = np.percentile(rgb[:, c], [2, 98])
        if hi - lo < 1e-6:
            rgb[:, c] = 0.5
        else:
            rgb[:, c] = np.clip((rgb[:, c] - lo) / (hi - lo), 0, 1)
    
    rgb_img = rgb.reshape(H, W, 3)
    if return_pca:
        return rgb_img, pca_model
    return rgb_img


def load_gt_features(feature_dir, scale, frame_ids):
    """加载 GT 特征。"""
    scale_dir = Path(feature_dir) / scale
    feats = {}
    for fpath in sorted(scale_dir.glob(f'rgb_*_{scale}_*.pt')):
        m = re.search(r'rgb_(\d+)_', fpath.name)
        if m:
            fid = int(m.group(1))
            if fid in frame_ids:
                feat = torch.load(str(fpath), map_location='cpu').float()
                feat = F.normalize(feat, p=2, dim=0)
                feats[fid] = feat
    return feats


def render_features(model, pose, fx, fy, cx, cy, h, w):
    """渲染一帧特征。"""
    result = FeatureRenderer.render_features(
        gaussian_model=model,
        viewmat=pose,
        fx=fx, fy=fy, cx=cx, cy=cy,
        img_height=h, img_width=w,
        feature_height=h, feature_width=w,
        norm_feat_before_render=True,
        norm_feat_after_render=False,
    )
    rendered = result['feature_map']  # [D, H, W]
    rendered = F.normalize(rendered, p=2, dim=0)
    return rendered


def make_comparison_grid(gt_feats, rendered_feats, frame_ids, scale, version, cos_scores):
    """
    生成对比图: 上行=GT, 下行=Rendered, 每列一帧。
    使用同一个 PCA model 确保颜色一致。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_frames = len(frame_ids)
    fig, axes = plt.subplots(2, n_frames, figsize=(4 * n_frames, 8))
    if n_frames == 1:
        axes = axes.reshape(2, 1)
    
    # 收集所有特征用于拟合统一的 PCA
    all_flat = []
    for fid in frame_ids:
        if fid in gt_feats and fid in rendered_feats:
            gt = gt_feats[fid]
            rd = rendered_feats[fid]
            D, H, W = gt.shape
            all_flat.append(gt.reshape(D, -1).T.cpu().numpy())
            all_flat.append(rd.reshape(D, -1).T.cpu().numpy())
    
    if not all_flat:
        print(f"  [WARN] 没有可用帧，跳过")
        plt.close()
        return None
    
    # 采样子集进行 PCA 拟合 (避免内存爆炸)
    combined = np.concatenate(all_flat, axis=0)
    if combined.shape[0] > 50000:
        indices = np.random.choice(combined.shape[0], 50000, replace=False)
        combined = combined[indices]
    
    pca_model = PCA(n_components=3)
    pca_model.fit(combined)
    
    for i, fid in enumerate(frame_ids):
        if fid not in gt_feats or fid not in rendered_feats:
            for row in range(2):
                axes[row, i].axis('off')
                axes[row, i].set_title(f"Frame {fid} N/A")
            continue
        
        gt_rgb = pca_colorize(gt_feats[fid], pca_model=pca_model)
        rd_rgb = pca_colorize(rendered_feats[fid], pca_model=pca_model)
        
        cos = cos_scores.get(fid, 0)
        
        axes[0, i].imshow(gt_rgb)
        axes[0, i].set_title(f"GT #{fid}", fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(rd_rgb)
        axes[1, i].set_title(f"Rendered #{fid}\ncos={cos:.3f}", fontsize=10)
        axes[1, i].axis('off')
    
    fig.suptitle(f"{version} / {scale}  (avg cos = {np.mean(list(cos_scores.values())):.4f})",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def main():
    device = torch.device('cuda')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载 poses
    traj = np.loadtxt(TRAJ_PATH).reshape(-1, 4, 4).astype(np.float32)
    poses_w2c = np.linalg.inv(traj).astype(np.float32)
    
    for version, model_base in MODEL_VERSIONS.items():
        for scale in ['coarse', 'mid', 'fine']:
            model_dir = Path(model_base) / scale
            ckpt_path = model_dir / 'best_model.pth'
            
            if not ckpt_path.exists():
                print(f"[SKIP] {version}/{scale}: {ckpt_path} 不存在")
                continue
            
            print(f"\n{'='*60}")
            print(f"  {version} / {scale}")
            print(f"{'='*60}")
            
            feat_dim = SCALE_DIMS[scale]
            
            # 加载模型
            model = GaussianFeatureModel(feature_dim=feat_dim)
            model.load_ply(str(PROJECT_ROOT / PLY_PATH))
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            model._loc_feature.data.copy_(ckpt['loc_feature'])
            model = model.to(device)
            
            # 加载 GT 特征
            gt_feats = load_gt_features(
                str(PROJECT_ROOT / FEATURE_DIR), scale, set(TEST_FRAMES)
            )
            
            # 获取渲染分辨率
            sample_fid = next(iter(gt_feats.keys()))
            feat_h, feat_w = gt_feats[sample_fid].shape[1], gt_feats[sample_fid].shape[2]
            
            # 计算渲染内参
            scale_x = feat_w / IMG_W
            scale_y = feat_h / IMG_H
            rfx = FX * scale_x
            rfy = FY * scale_y
            rcx = CX * scale_x
            rcy = CY * scale_y
            
            # 渲染并计算 cosine
            rendered_feats = {}
            cos_scores = {}
            
            for fid in TEST_FRAMES:
                if fid not in gt_feats or fid >= len(poses_w2c):
                    continue
                pose = torch.tensor(poses_w2c[fid], device=device)
                
                with torch.no_grad():
                    rendered = render_features(model, pose, rfx, rfy, rcx, rcy, feat_h, feat_w)
                
                rendered_feats[fid] = rendered.cpu()
                cos = F.cosine_similarity(
                    rendered.reshape(feat_dim, -1).T,
                    gt_feats[fid].to(device).reshape(feat_dim, -1).T,
                    dim=1
                ).mean().item()
                cos_scores[fid] = cos
                print(f"  Frame {fid}: cos={cos:.4f}")
            
            avg_cos = np.mean(list(cos_scores.values()))
            print(f"  → avg cos = {avg_cos:.4f}")
            
            # 生成对比图
            fig = make_comparison_grid(gt_feats, rendered_feats, TEST_FRAMES, scale, version, cos_scores)
            if fig is not None:
                out_path = Path(OUTPUT_DIR) / f"{version}_{scale}_comparison.png"
                fig.savefig(str(PROJECT_ROOT / out_path), dpi=120, bbox_inches='tight')
                import matplotlib.pyplot as plt
                plt.close(fig)
                print(f"  → 保存: {out_path}")
            
            # 清理 GPU
            del model
            torch.cuda.empty_cache()
    
    print(f"\n所有对比图已保存到 {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
