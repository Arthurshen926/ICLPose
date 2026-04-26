#!/usr/bin/env python3
"""
多尺度特征渲染可视化 (拆分 fine_sd / fine_dino)
================================================
从训练好的 checkpoint 渲染特征图，与 GT 逐尺度对比。
5 列: fine_sd(64d) | fine_dino(64d) | mid(64d) | coarse(32d) | fine_combined(128d)
2 行: Rendered | GT
"""
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from feature_3dgs.multiscale_gaussian_model import MultiScaleGaussianModel
from feature_3dgs.feature_renderer import FeatureRenderer
from feature_3dgs.multiscale_dataset import MultiScaleFeatureDataset


def feat_to_rgb(feat_chw, shared_pca=None):
    """[C,H,W] → [H,W,3] (PCA or shared PCA)"""
    C, H, W = feat_chw.shape
    flat = feat_chw.reshape(C, -1).T  # [HW, C]
    if shared_pca is not None:
        rgb = shared_pca.transform(flat)
    else:
        pca = PCA(n_components=3)
        rgb = pca.fit_transform(flat)
    rgb = (rgb - rgb.min(axis=0)) / (rgb.max(axis=0) - rgb.min(axis=0) + 1e-8)
    return rgb.reshape(H, W, 3)


def render_frame(model, pose, fine_fx, fine_fy, fine_cx, fine_cy,
                 fine_H, fine_W, mid_H, mid_W, coarse_H, coarse_W):
    """渲染 + 拆分 + downsample + normalize"""
    result = FeatureRenderer.render_features(
        gaussian_model=model, viewmat=pose,
        fx=fine_fx, fy=fine_fy, cx=fine_cx, cy=fine_cy,
        img_height=fine_H, img_width=fine_W,
        norm_feat_before_render=True, norm_feat_after_render=False,
    )
    fm = result['feature_map']  # [224, H, W]
    S = MultiScaleGaussianModel

    fine_sd = F.normalize(fm[S.FINE_SD_START:S.FINE_SD_END], p=2, dim=0)
    fine_dino = F.normalize(fm[S.FINE_DINO_START:S.FINE_DINO_END], p=2, dim=0)
    fine_combined = F.normalize(fm[S.FINE_SD_START:S.FINE_END], p=2, dim=0)

    mid = F.interpolate(
        fm[S.MID_START:S.MID_END].unsqueeze(0),
        size=(mid_H, mid_W), mode='bilinear', align_corners=False
    ).squeeze(0)
    mid = F.normalize(mid, p=2, dim=0)

    coarse = F.interpolate(
        fm[S.COARSE_START:S.COARSE_END].unsqueeze(0),
        size=(coarse_H, coarse_W), mode='bilinear', align_corners=False
    ).squeeze(0)
    coarse = F.normalize(coarse, p=2, dim=0)

    return {
        'fine_sd': fine_sd, 'fine_dino': fine_dino,
        'fine_combined': fine_combined,
        'mid': mid, 'coarse': coarse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='best_model.pth or final_model.pth')
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--traj_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output/vis_multiscale')
    parser.add_argument('--frames', type=str, default='0,100,250,450,700',
                        help='要可视化的帧 ID')
    parser.add_argument('--fx', type=float, default=320.0)
    parser.add_argument('--fy', type=float, default=320.0)
    parser.add_argument('--cx', type=float, default=319.5)
    parser.add_argument('--cy', type=float, default=239.5)
    args = parser.parse_args()

    device = torch.device('cuda')

    # 加载模型
    model = MultiScaleGaussianModel()
    model.load_ply(args.ply_path)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model._loc_feature.data.copy_(ckpt['loc_feature'])
    model = model.to(device)
    print(f"模型加载完成: iter={ckpt.get('iteration','?')}, loss={ckpt.get('loss','?'):.6f}")

    # 加载数据集
    dataset = MultiScaleFeatureDataset(
        feature_dir=args.feature_dir,
        traj_path=args.traj_path,
        normalize_features=True,
    )
    fine_H, fine_W = dataset.fine_hw
    mid_H, mid_W = dataset.mid_hw
    coarse_H, coarse_W = dataset.coarse_hw

    scale_x, scale_y = fine_W / 640, fine_H / 480
    fine_fx = args.fx * scale_x
    fine_fy = args.fy * scale_y
    fine_cx = args.cx * scale_x
    fine_cy = args.cy * scale_y

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = [int(x) for x in args.frames.split(',')]
    S = MultiScaleGaussianModel

    for fid in frame_ids:
        try:
            idx = dataset.frame_ids.index(fid)
        except ValueError:
            print(f"  帧 {fid} 不在数据集中，跳过")
            continue

        sample = dataset[idx]
        gt_fine = sample['fine_feat'].to(device)      # [128, 35, 46]
        gt_mid = sample['mid_feat'].to(device)        # [64, 15, 20]
        gt_coarse = sample['coarse_feat'].to(device)  # [32, 7, 10]
        pose = sample['pose'].to(device)

        # GT 拆分 fine_sd / fine_dino
        gt_fine_sd = gt_fine[:64]
        gt_fine_dino = gt_fine[64:]

        with torch.no_grad():
            rendered = render_frame(
                model, pose, fine_fx, fine_fy, fine_cx, fine_cy,
                fine_H, fine_W, mid_H, mid_W, coarse_H, coarse_W
            )

        # 5 列可视化
        cols = [
            ('Fine SD\n(64d)', rendered['fine_sd'], gt_fine_sd),
            ('Fine DINO\n(64d)', rendered['fine_dino'], gt_fine_dino),
            ('Mid SD-s4\n(64d)', rendered['mid'], gt_mid),
            ('Coarse SD-s5\n(32d)', rendered['coarse'], gt_coarse),
            ('Fine Combined\n(SD+DINO 128d)', rendered['fine_combined'], gt_fine),
        ]

        fig, axes = plt.subplots(2, 5, figsize=(30, 10))
        for j, (title, r_tensor, g_tensor) in enumerate(cols):
            r_np = r_tensor.cpu().numpy()
            g_np = g_tensor.cpu().numpy()

            # 用 GT 的 PCA fit，rendered 用相同的变换
            C = g_np.shape[0]
            pca = PCA(n_components=3)
            gt_flat = g_np.reshape(C, -1).T
            pca.fit(gt_flat)

            r_rgb = feat_to_rgb(r_np, shared_pca=pca)
            g_rgb = feat_to_rgb(g_np, shared_pca=pca)

            axes[0, j].imshow(r_rgb)
            axes[0, j].set_title(f'Rendered {title}', fontsize=11)
            axes[0, j].axis('off')

            axes[1, j].imshow(g_rgb)
            axes[1, j].set_title(f'GT {title}', fontsize=11)
            axes[1, j].axis('off')

        fig.suptitle(f'Frame {fid} - Multi-Scale Feature Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        save_path = out_dir / f'frame_{fid:04d}_multiscale.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Frame {fid}: {save_path}")

    print(f"\n可视化保存至: {out_dir}")


if __name__ == '__main__':
    main()
