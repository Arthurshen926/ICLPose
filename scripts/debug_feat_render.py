"""
Debug: compare checkpoint vs PLY feature rendering, and match training vis output.
"""
import sys, os, torch, numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from feature_3dgs.multiscale_gaussian_model import MultiScaleGaussianModel
from feature_3dgs.feature_renderer import FeatureRenderer

S = MultiScaleGaussianModel

# === Training-style PCA (sklearn, global min-max) ===
def train_pca(feat_chw_np):
    """Exact copy of training's feat_to_rgb: [C,H,W] np → [H,W,3] np"""
    C, H, W = feat_chw_np.shape
    flat = feat_chw_np.reshape(C, -1).T  # [H*W, C]
    if flat.shape[0] < 3:
        return np.zeros((H, W, 3))
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(flat)  # [H*W, 3]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb.reshape(H, W, 3)


def main():
    device = torch.device('cuda:0')
    out_dir = 'output/feature_3dgs/room_0_vis'
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load from PLY ──
    print("Loading PLY model...")
    ply_model = MultiScaleGaussianModel()
    ply_model.load_ply_with_features('output/feature_3dgs/room_0_v1/point_cloud_with_features.ply')
    ply_model = ply_model.to(device)
    ply_feat = ply_model._loc_feature.clone()
    print(f"  PLY: {ply_model.num_gaussians} Gaussians, feat shape={ply_feat.shape}")
    print(f"  PLY feat: min={ply_feat.min():.4f}, max={ply_feat.max():.4f}, mean={ply_feat.mean():.4f}")

    # ── 2. Load from checkpoint ──
    print("\nLoading checkpoint model...")
    ckpt = torch.load('output/feature_3dgs/room_0_v1/final_model.pth', map_location='cpu')
    ckpt_feat = ckpt['loc_feature']
    print(f"  Ckpt feat: shape={ckpt_feat.shape}, min={ckpt_feat.min():.4f}, max={ckpt_feat.max():.4f}, mean={ckpt_feat.mean():.4f}")

    # ── 3. Compare features ──
    diff = (ply_feat.cpu() - ckpt_feat).abs()
    print(f"\n  Feature diff: max={diff.max():.8f}, mean={diff.mean():.8f}")
    if diff.max() > 1e-4:
        print("  ⚠️ WARNING: PLY and checkpoint features differ significantly!")
        # Find where they differ most
        max_idx = diff.sum(dim=1).argmax()
        print(f"    Worst Gaussian idx={max_idx}")
        print(f"    PLY:  {ply_feat[max_idx, :5].cpu().tolist()}")
        print(f"    Ckpt: {ckpt_feat[max_idx, :5].tolist()}")
    else:
        print("  ✓ PLY and checkpoint features match!")

    # ── 4. Also compare geometry ──
    ply_xyz = ply_model._xyz.clone().cpu()
    # Load the geometry-only model used for training
    geo_model = MultiScaleGaussianModel()
    geo_model.load_ply('output/2dgs_models/room_0/v8_fixed_poses/point_cloud/iteration_30000/point_cloud.ply')
    geo_xyz = geo_model._xyz.clone().cpu()
    if ply_xyz.shape == geo_xyz.shape:
        xyz_diff = (ply_xyz - geo_xyz).abs().max()
        print(f"\n  Geometry xyz diff: max={xyz_diff:.8f}")
    else:
        print(f"\n  Geometry shape mismatch: PLY={ply_xyz.shape}, Geo={geo_xyz.shape}")

    # ── 5. Load pose (frame 0, same as training vis) ──
    raw = np.loadtxt('dataset/room_0/Sequence_1/traj_w_c.txt').reshape(-1, 4, 4)
    c2w = raw[0]
    R, t = c2w[:3, :3], c2w[:3, 3]
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)
    print(f"\n  Frame 0 w2c:\n{w2c}")

    # ── 6. Exact training intrinsics (defaults from train_multiscale_embedding_v2.py) ──
    fx, fy = 320.0, 320.0
    cx, cy = 319.5, 239.5
    ref_W, ref_H = 640, 480

    # Read actual resolutions from compressed features
    comp_dir = 'output/features_multiscale_compressed/room_0'
    fine_t = torch.load(os.path.join(comp_dir, 'fine_sd', sorted(os.listdir(os.path.join(comp_dir, 'fine_sd')))[0]), map_location='cpu')
    mid_t = torch.load(os.path.join(comp_dir, 'mid', sorted(os.listdir(os.path.join(comp_dir, 'mid')))[0]), map_location='cpu')
    coarse_t = torch.load(os.path.join(comp_dir, 'coarse', sorted(os.listdir(os.path.join(comp_dir, 'coarse')))[0]), map_location='cpu')
    fine_H, fine_W = fine_t.shape[1], fine_t.shape[2]
    mid_H, mid_W = mid_t.shape[1], mid_t.shape[2]
    coarse_H, coarse_W = coarse_t.shape[1], coarse_t.shape[2]
    print(f"\n  Resolutions: fine={fine_W}×{fine_H}, mid={mid_W}×{mid_H}, coarse={coarse_W}×{coarse_H}")

    # Training's _scale_intrinsics
    def scale_intr(H, W):
        return {
            'fx': fx * W / ref_W,
            'fy': fy * H / ref_H,
            'cx': cx * W / ref_W,
            'cy': cy * H / ref_H,
            'H': H, 'W': W,
        }

    scale_intrinsics = {
        'fine': scale_intr(fine_H, fine_W),
        'mid': scale_intr(mid_H, mid_W),
        'coarse': scale_intr(coarse_H, coarse_W),
    }

    for k, v in scale_intrinsics.items():
        print(f"  {k}: fx={v['fx']:.4f}, fy={v['fy']:.4f}, cx={v['cx']:.4f}, cy={v['cy']:.4f}, H={v['H']}, W={v['W']}")

    # Compare with what vis script would compute
    print("\n  === Vis script intrinsics comparison ===")
    for k, v in scale_intrinsics.items():
        vis_cx = v['W'] / 2.0
        vis_cy = v['H'] / 2.0
        print(f"  {k}: training cx={v['cx']:.4f}, vis cx={vis_cx:.4f}, diff={abs(v['cx']-vis_cx):.4f}")
        print(f"  {k}: training cy={v['cy']:.4f}, vis cy={vis_cy:.4f}, diff={abs(v['cy']-vis_cy):.4f}")

    # ── 7. Render per scale using EXACT training method ──
    print("\n  Rendering with training method...")
    raw_feat = ply_model._loc_feature

    scale_renders = {}
    for sname, ch_start, ch_end in [('fine', S.FINE_SD_START, S.FINE_END),
                                     ('mid', S.MID_START, S.MID_END),
                                     ('coarse', S.COARSE_START, S.COARSE_END)]:
        si = scale_intrinsics[sname]
        colors = F.normalize(raw_feat[:, ch_start:ch_end], p=2, dim=-1)

        result = FeatureRenderer.render_features(
            gaussian_model=ply_model, viewmat=viewmat,
            fx=si['fx'], fy=si['fy'], cx=si['cx'], cy=si['cy'],
            img_height=si['H'], img_width=si['W'],
            norm_feat_before_render=False, norm_feat_after_render=False,
            colors_override=colors,
        )
        fm = F.normalize(result['feature_map'], p=2, dim=0)
        alpha = result['alpha']
        scale_renders[sname] = fm
        print(f"  {sname}: shape={fm.shape}, alpha mean={alpha.mean():.4f}, "
              f"feat norm mean={fm.norm(dim=0).mean():.4f}")

    # ── 8. PCA colorize using EXACT training method (sklearn) ──
    print("\n  PCA colorizing (sklearn, same as training)...")
    pca_imgs = {}
    for sname, fm in scale_renders.items():
        pca_imgs[sname] = train_pca(fm.detach().cpu().numpy())

    # ── 9. Also render using vis script method (cx=W/2, cy=H/2) ──
    print("\n  Rendering with vis script method...")
    scale_renders_vis = {}
    for sname, ch_start, ch_end in [('fine', S.FINE_SD_START, S.FINE_END),
                                     ('mid', S.MID_START, S.MID_END),
                                     ('coarse', S.COARSE_START, S.COARSE_END)]:
        si = scale_intrinsics[sname]
        vis_cx = si['W'] / 2.0
        vis_cy = si['H'] / 2.0
        colors = F.normalize(raw_feat[:, ch_start:ch_end], p=2, dim=-1)

        result = FeatureRenderer.render_features(
            gaussian_model=ply_model, viewmat=viewmat,
            fx=si['fx'], fy=si['fy'], cx=vis_cx, cy=vis_cy,
            img_height=si['H'], img_width=si['W'],
            norm_feat_before_render=False, norm_feat_after_render=False,
            colors_override=colors,
        )
        fm = F.normalize(result['feature_map'], p=2, dim=0)
        scale_renders_vis[sname] = fm

    pca_imgs_vis = {}
    for sname, fm in scale_renders_vis.items():
        pca_imgs_vis[sname] = train_pca(fm.detach().cpu().numpy())

    # ── 10. Load GT features for frame 0 and PCA ──
    print("\n  Loading GT features for comparison...")
    gt_pcas = {}
    # Find actual filenames (they include dimensions in name)
    fine_sd_files = sorted(os.listdir(os.path.join(comp_dir, 'fine_sd')))
    fine_dino_files = sorted(os.listdir(os.path.join(comp_dir, 'fine_dino')))
    mid_files = sorted(os.listdir(os.path.join(comp_dir, 'mid')))
    coarse_files = sorted(os.listdir(os.path.join(comp_dir, 'coarse')))
    # Frame 0 files
    f0_sd = [f for f in fine_sd_files if f.startswith('rgb_0_')][0]
    f0_dino = [f for f in fine_dino_files if f.startswith('rgb_0_')][0]
    f0_mid = [f for f in mid_files if f.startswith('rgb_0_')][0]
    f0_coarse = [f for f in coarse_files if f.startswith('rgb_0_')][0]
    gt_fine_sd = torch.load(os.path.join(comp_dir, 'fine_sd', f0_sd), map_location='cpu')
    gt_fine_dino = torch.load(os.path.join(comp_dir, 'fine_dino', f0_dino), map_location='cpu')
    gt_fine = torch.cat([gt_fine_sd, gt_fine_dino], dim=0)  # [128, H, W]
    gt_mid = torch.load(os.path.join(comp_dir, 'mid', f0_mid), map_location='cpu')
    gt_coarse = torch.load(os.path.join(comp_dir, 'coarse', f0_coarse), map_location='cpu')
    print(f"  GT fine: {gt_fine.shape}, mid: {gt_mid.shape}, coarse: {gt_coarse.shape}")

    gt_pcas['fine'] = train_pca(gt_fine.numpy())
    gt_pcas['mid'] = train_pca(gt_mid.numpy())
    gt_pcas['coarse'] = train_pca(gt_coarse.numpy())

    # ── 11. Load training vis image for visual comparison ──
    from PIL import Image
    training_vis = np.array(Image.open('output/feature_3dgs/room_0_v1/vis/final_frame0.png'))

    # ── 12. Plot comprehensive comparison ──
    # 5 rows: training_vis | GT PCA | rendered PCA (training intr) | rendered PCA (vis intr) | diff
    fig, axes = plt.subplots(5, 3, figsize=(15, 22))
    scales = ['fine', 'mid', 'coarse']

    # Row 0: Training vis image (spans all 3 cols)
    for j in range(3):
        axes[0, j].axis('off')
    axes[0, 1].imshow(training_vis)
    axes[0, 1].set_title('Training visualization (final_frame0.png)', fontsize=12)
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')

    for j, sname in enumerate(scales):
        # Row 1: GT
        axes[1, j].imshow(gt_pcas[sname])
        axes[1, j].set_title(f'GT {sname}', fontsize=11)
        axes[1, j].axis('off')

        # Row 2: Rendered (training intrinsics)
        axes[2, j].imshow(pca_imgs[sname])
        axes[2, j].set_title(f'Rendered (training intr) {sname}', fontsize=11)
        axes[2, j].axis('off')

        # Row 3: Rendered (vis script intrinsics)
        axes[3, j].imshow(pca_imgs_vis[sname])
        axes[3, j].set_title(f'Rendered (vis intr) {sname}', fontsize=11)
        axes[3, j].axis('off')

        # Row 4: Pixel diff between rendered and GT
        diff = np.abs(pca_imgs[sname] - gt_pcas[sname])
        axes[4, j].imshow(diff, vmin=0, vmax=0.5)
        axes[4, j].set_title(f'|Rendered - GT| {sname}\nmax={diff.max():.3f}', fontsize=11)
        axes[4, j].axis('off')

    row_labels = ['Training Vis', 'GT PCA', 'Rendered (training intr)', 'Rendered (vis intr)', '|Rendered - GT|']
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=10)

    fig.suptitle('Room_0 Feature Rendering Diagnostic\n'
                 'Compare: GT features vs Rendered from 3DGS', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'debug_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")

    # ── 13. Feature-level statistics comparison ──
    print("\n  === Per-scale feature statistics ===")
    for sname in scales:
        rendered = scale_renders[sname].detach().cpu()
        if sname == 'fine':
            gt = F.normalize(gt_fine.float(), p=2, dim=0)
        elif sname == 'mid':
            gt = F.normalize(gt_mid.float(), p=2, dim=0)
        else:
            gt = F.normalize(gt_coarse.float(), p=2, dim=0)

        cosine_sim = (rendered * gt).sum(dim=0)
        print(f"  {sname}:")
        print(f"    Rendered: shape={rendered.shape}, norm={rendered.norm(dim=0).mean():.4f}")
        print(f"    GT:       shape={gt.shape}, norm={gt.norm(dim=0).mean():.4f}")
        print(f"    Cosine similarity: mean={cosine_sim.mean():.4f}, min={cosine_sim.min():.4f}")
        print(f"    L2 diff: mean={(rendered - gt).norm(dim=0).mean():.4f}")


if __name__ == '__main__':
    main()
