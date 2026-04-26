#!/usr/bin/env python3
"""
快速验证 2DGS 特征重建质量
比较 v2 vs v3 的 coarse 特征渲染效果 (CPU 模式，不干扰 GPU 训练)
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import re

# 使用 GPU 5 (显存最空闲)，避免干扰其他进程
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')

sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 路径配置
PLY      = 'output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply'
FEAT_DIR = Path('output/features_da3/OldHospital_indexed/coarse')
CKPT_V3  = 'output/feature_3dgs/oldhospital_da3_perscale_v3/best_model.pth'
CKPT_V2  = 'output/feature_3dgs/oldhospital_da3_perscale_v2/coarse/best_model.pth'
TRAJ     = 'output/features_da3/OldHospital_indexed/traj_w_c.txt'
OUT      = 'output/da3_perscale_vis_v3/v3_vs_v2_verify.png'

def load_model(ckpt_path, feat_dim=32):
    model = GaussianFeatureModel(feature_dim=feat_dim)
    model.load_ply(PLY)
    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model._loc_feature.data.copy_(ckpt['loc_feature'])
    model.eval()
    cos = ckpt.get('cosine_similarity', float('nan'))
    print(f"  loaded {ckpt_path} (cos={cos:.4f})")
    return model, cos

def pca_rgb(tensors):
    """几个 [C,H,W] tensor → PCA 3ch, 归一到 [0,1]"""
    C = tensors[0].shape[0]
    all_flat = torch.cat([t.reshape(C, -1).T for t in tensors], dim=0)  # [N*HW, C]
    mean = all_flat.mean(0, keepdim=True)
    centered = all_flat - mean
    _, _, V = torch.linalg.svd(centered, full_matrices=False)
    proj = centered @ V[:3].T                                             # [N*HW, 3]
    pmin = proj.min(0, keepdim=True)[0]
    pmax = proj.max(0, keepdim=True)[0]
    proj = ((proj - pmin) / (pmax - pmin + 1e-8)).clamp(0, 1)
    out = []
    hw = tensors[0].shape[1] * tensors[0].shape[2]
    for i, t in enumerate(tensors):
        H, W = t.shape[1], t.shape[2]
        out.append(proj[i*hw:(i+1)*hw].reshape(H, W, 3).numpy())
    return out


if __name__ == '__main__':
    print('[1] 加载 v2/v3 模型...')
    model_v3, cos_v3_best = load_model(CKPT_V3)
    model_v2, cos_v2_best = load_model(CKPT_V2)

    print('[2] 加载轨迹...')
    traj = np.loadtxt(TRAJ).reshape(-1, 4, 4).astype(np.float32)
    poses = np.linalg.inv(traj)

    feat_files = sorted(FEAT_DIR.glob('rgb_*_coarse_*.pt'))
    sample_ids = [0, len(feat_files)//4, len(feat_files)//2]
    feat_files = [feat_files[i] for i in sample_ids]

    feat_h, feat_w = 15, 26
    renderer_args = dict(
        fx=1663.12 * feat_w / 1920,
        fy=1663.12 * feat_h / 1080,
        cx=960.0 * feat_w / 1920,
        cy=540.0 * feat_h / 1080,
        img_height=feat_h, img_width=feat_w,
        feature_height=feat_h, feature_width=feat_w,
        norm_feat_before_render=True,
        norm_feat_after_render=False,
    )

    print('[3] 渲染 3 帧...')
    fig, axes = plt.subplots(len(feat_files), 3, figsize=(12, 4 * len(feat_files)))

    for row, fpath in enumerate(feat_files):
        m = re.search(r'rgb_(\d+)_', fpath.name)
        fid = int(m.group(1))
        gt_feat = torch.load(str(fpath), map_location='cpu').float()
        gt_feat = F.normalize(gt_feat, p=2, dim=0)
        pose = torch.tensor(poses[fid])

        with torch.no_grad():
            r3 = FeatureRenderer.render_features(gaussian_model=model_v3, viewmat=pose, **renderer_args)
            rv3 = F.normalize(r3['feature_map'], p=2, dim=0)
            r2 = FeatureRenderer.render_features(gaussian_model=model_v2, viewmat=pose, **renderer_args)
            rv2 = F.normalize(r2['feature_map'], p=2, dim=0)

        cos_v2 = F.cosine_similarity(rv2, gt_feat, dim=0).mean().item()
        cos_v3 = F.cosine_similarity(rv3, gt_feat, dim=0).mean().item()
        print(f"  frame {fid}: v2_cos={cos_v2:.3f}  v3_cos={cos_v3:.3f}  Δ={cos_v3-cos_v2:+.3f}")

        gt_img, v2_img, v3_img = pca_rgb([gt_feat, rv2, rv3])

        axes[row][0].imshow(gt_img)
        axes[row][0].set_title(f'GT (frame {fid})', fontsize=10)
        axes[row][0].axis('off')
        axes[row][1].imshow(v2_img)
        axes[row][1].set_title(f'v2 (cos={cos_v2:.3f})', fontsize=10)
        axes[row][1].axis('off')
        axes[row][2].imshow(v3_img)
        axes[row][2].set_title(f'v3 (cos={cos_v3:.3f})', fontsize=10)
        axes[row][2].axis('off')

    plt.suptitle(f'DA3 Coarse Feats: GT vs v2(best={cos_v2_best:.4f}) vs v3(best={cos_v3_best:.4f})', fontsize=13)
    plt.tight_layout()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[4] 保存: {OUT}')
