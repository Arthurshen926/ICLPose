"""Visualize DA3-compressed vs 3DGS-rendered vs depth-warped features.
Generates PCA + cosine similarity heatmaps."""
import sys, yaml, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, '/root/ICLPose')

from modules.depth_warp import backward_warp_features
from modules.multiscale_renderer import MultiScaleRenderer
from data.dataset_v4 import PoseDatasetV4, perturb_pose
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pca_colorize(feat_chw, n_components=3):
    """(C,H,W) → (H,W,3) RGB via PCA, scaled to [0,1]."""
    C, H, W = feat_chw.shape
    flat = feat_chw.reshape(C, -1).T.cpu().numpy()  # (N, C)
    pca = PCA(n_components=n_components)
    rgb = pca.fit_transform(flat)  # (N, 3)
    rgb -= rgb.min(0)
    mx = rgb.max(0)
    mx[mx == 0] = 1
    rgb /= mx
    return rgb.reshape(H, W, 3)


def cosine_sim_map(a, b):
    """Pixel-wise cosine sim between (C,H,W) tensors → (H,W) numpy."""
    return F.cosine_similarity(a.float(), b.float(), dim=0).cpu().numpy()


def self_sim_map(feat_chw, center_y=None, center_x=None):
    """Cosine similarity of every pixel to a chosen center pixel → (H,W)."""
    C, H, W = feat_chw.shape
    if center_y is None:
        center_y, center_x = H // 2, W // 2
    anchor = feat_chw[:, center_y, center_x].reshape(C, 1, 1)
    return F.cosine_similarity(feat_chw.float(), anchor.float(), dim=0).cpu().numpy()


def main():
    with open('configs/exp188_oh_depth_warp.yaml') as f:
        cfg = yaml.safe_load(f)

    ds = PoseDatasetV4(
        feature_base_dir=cfg['data']['train_feature_dir'],
        traj_path=cfg['data']['train_traj_path'],
        noise_rot_deg=2.0, noise_trans_m=0.05, is_train=False,
    )
    sample = ds[100]  # pick a frame with some structure
    query_feats = {k: v.unsqueeze(0).cuda() for k, v in sample['query_feats'].items()}
    pose_gt = sample['pose_gt'].unsqueeze(0).cuda()
    pose_noisy = perturb_pose(sample['pose_gt'].clone(), 5.0, 0.1).unsqueeze(0).cuda()

    renderer = MultiScaleRenderer(**cfg['renderer'], device='cuda')
    scale_intrinsics = renderer.get_scale_intrinsics()

    depth_noisy = renderer.render_depth_batch(pose_noisy)
    depth_gt = renderer.render_depth_batch(pose_gt)

    warped = backward_warp_features(
        {k: v.float() for k, v in query_feats.items()},
        depth_noisy.float(), pose_gt.float(), pose_noisy.float(), scale_intrinsics)

    warped_gt = backward_warp_features(
        {k: v.float() for k, v in query_feats.items()},
        depth_gt.float(), pose_gt.float(), pose_gt.float(), scale_intrinsics)

    gs_out = renderer.render_batch(pose_gt, return_depth=False)
    gs_feats = {k.replace('_feat', ''): v for k, v in gs_out.items()}

    # ===== Figure 1: PCA visualization per scale =====
    fig1, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig1.suptitle('PCA Feature Visualization (frame 100)', fontsize=16, y=0.98)
    col_titles = ['DA3 Query (disk)', 'DA3 Warp@GT', 'DA3 Warp@5°/0.1m', '3DGS Render@GT']
    for i, scale in enumerate(['coarse', 'mid', 'fine_sd', 'fine_dino']):
        q = query_feats[scale][0]
        wg = warped_gt[scale][0]
        wn = warped[scale][0]
        gs = gs_feats[scale][0]

        for j, (feat, title) in enumerate(zip([q, wg, wn, gs], col_titles)):
            ax = axes[i, j]
            ax.imshow(pca_colorize(feat))
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(f'{scale}\n{feat.shape[0]}d×{feat.shape[1]}×{feat.shape[2]}',
                             fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig('output/feature_pca_comparison.png', dpi=150, bbox_inches='tight')
    print('Saved: output/feature_pca_comparison.png')

    # ===== Figure 2: Cosine similarity heatmaps =====
    fig2, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig2.suptitle('Pixel-wise Cosine Similarity Maps', fontsize=16, y=0.98)
    col_titles2 = ['Warp@GT vs Query\n(should be 1.0)',
                   'Warp@5° vs Query\n(depth-warp signal)',
                   '3DGS@GT vs Query\n(GS domain gap)']
    for i, scale in enumerate(['coarse', 'mid', 'fine_sd', 'fine_dino']):
        q = query_feats[scale][0]
        wg = warped_gt[scale][0]
        wn = warped[scale][0]
        gs = gs_feats[scale][0]

        sim_wg = cosine_sim_map(q, wg)
        sim_wn = cosine_sim_map(q, wn)
        sim_gs = cosine_sim_map(q, gs)

        for j, (sim, title) in enumerate(zip([sim_wg, sim_wn, sim_gs], col_titles2)):
            ax = axes[i, j]
            im = ax.imshow(sim, vmin=0.7, vmax=1.0, cmap='RdYlGn')
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(f'{scale}', fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig('output/feature_cosine_similarity.png', dpi=150, bbox_inches='tight')
    print('Saved: output/feature_cosine_similarity.png')

    # ===== Figure 3: Self-similarity — the CRITICAL diagnostic =====
    fig3, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig3.suptitle('Self-Similarity: Center Pixel vs All Others\n(Low values = discriminative, High = everything looks the same)',
                  fontsize=14, y=0.98)
    col_titles3 = ['DA3 Compressed (32/64d)', '3DGS Rendered (32/64d)']
    for i, scale in enumerate(['coarse', 'mid', 'fine_sd', 'fine_dino']):
        q = query_feats[scale][0]
        gs = gs_feats[scale][0]
        H, W = q.shape[1], q.shape[2]

        for j, (feat, title) in enumerate(zip([q, gs], col_titles3)):
            ax = axes[i, j]
            ssim = self_sim_map(feat, H // 2, W // 2)
            im = ax.imshow(ssim, vmin=0.7, vmax=1.0, cmap='hot')
            ax.plot(W // 2, H // 2, 'c+', markersize=15, markeredgewidth=2)
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(f'{scale}', fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    fig3.savefig('output/feature_self_similarity.png', dpi=150, bbox_inches='tight')
    print('Saved: output/feature_self_similarity.png')

    # ===== Summary statistics =====
    print('\n========================================')
    print('DIAGNOSTIC SUMMARY')
    print('========================================')
    for scale in ['coarse', 'mid', 'fine_sd', 'fine_dino']:
        q = query_feats[scale][0].float()
        gs = gs_feats[scale][0].float()
        C, H, W = q.shape

        # Self-similarity (DA3)
        q_flat = F.normalize(q.reshape(C, -1), dim=0)
        gram = q_flat.T @ q_flat
        mask = ~torch.eye(H * W, device=gram.device, dtype=torch.bool)
        selfsim = gram[mask]

        # GS domain gap
        gs_cos = F.cosine_similarity(q, gs, dim=0)

        print(f'\n{scale} [{C}d × {H}×{W}]:')
        print(f'  DA3 self-similarity:  mean={selfsim.mean():.4f}  std={selfsim.std():.4f}  '
              f'min={selfsim.min():.4f}  max={selfsim.max():.4f}')
        print(f'  GS vs Query cosine:   mean={gs_cos.mean():.4f}  std={gs_cos.std():.4f}  '
              f'min={gs_cos.min():.4f}')
        margin = gs_cos.mean() - selfsim.mean()
        print(f'  MATCHING MARGIN:      {margin:.4f}  '
              f'(>0 = correct pixel scores higher than avg random, NEGATIVE = BROKEN)')


if __name__ == '__main__':
    main()
