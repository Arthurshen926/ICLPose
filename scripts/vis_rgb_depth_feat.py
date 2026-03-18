"""
RGB / Depth / Multi-scale Feature PCA visualization for room_0 and OldHospital.
- RGB + Depth from geometry PLY
- Per-scale Feature PCA (fine_sd / fine_dino / mid / coarse) from feature PLY
- OldHospital: cameras.json uses R_c2w + position convention
- Room_0: traj_w_c.txt uses 4×4 c2w matrices
"""
import sys, os, json, torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.multiscale_gaussian_model import MultiScaleGaussianModel
from feature_3dgs.feature_renderer import FeatureRenderer

S = MultiScaleGaussianModel  # shorthand for scale constants


# ── Pose helpers ──────────────────────────────────────────

def load_traj_w2c(traj_path):
    """Load c2w poses from traj_w_c.txt → return list of w2c [4,4] np arrays."""
    raw = np.loadtxt(traj_path).reshape(-1, 4, 4)
    w2c_list = []
    for c2w in raw:
        R, t = c2w[:3, :3], c2w[:3, 3]
        w2c = np.eye(4, dtype=c2w.dtype)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -R.T @ t
        w2c_list.append(w2c)
    return w2c_list


def load_cameras_json_w2c(json_path):
    """Load cameras.json (3DGS format) → return list of w2c [4,4] np arrays.
    cameras.json convention:
        rotation = R_c2w (camera-to-world rotation)
        position = camera center in world
    So:  R_w2c = R_c2w^T,  t_w2c = -R_w2c @ position
    """
    with open(json_path) as f:
        cams = json.load(f)
    w2c_list = []
    for cam in cams:
        R_c2w = np.array(cam['rotation'])
        pos = np.array(cam['position'])
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ pos
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        w2c_list.append(w2c)
    return w2c_list, cams


# ── PCA colorization ─────────────────────────────────────

def pca_colorize(feat_chw):
    """[C,H,W] tensor/ndarray → [H,W,3] numpy RGB via sklearn PCA (same as training vis)."""
    from sklearn.decomposition import PCA
    if isinstance(feat_chw, torch.Tensor):
        feat_chw = feat_chw.cpu().numpy()
    C, H, W = feat_chw.shape
    flat = feat_chw.reshape(C, -1).T  # [HW, C]
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(flat)  # [HW, 3]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb.reshape(H, W, 3)


# ── Per-scale feature rendering (native resolution, colors_override) ──

# Scale definitions: (label, ch_start, ch_end, resolution_key)
SCALE_DEFS = [
    ('Fine (SD+DINO 128d)', S.FINE_SD_START, S.FINE_END, 'fine'),
    ('Mid SD-s4 (64d)',     S.MID_START,     S.MID_END,  'mid'),
    ('Coarse SD-s5 (32d)',  S.COARSE_START,  S.COARSE_END, 'coarse'),
]


def detect_scale_resolutions(compressed_feat_dir):
    """Auto-detect per-scale (H, W) from compressed feature files."""
    resolutions = {}
    mapping = {'fine': 'fine_sd', 'mid': 'mid', 'coarse': 'coarse'}
    for key, subdir in mapping.items():
        d = os.path.join(compressed_feat_dir, subdir)
        if os.path.exists(d):
            f = sorted(os.listdir(d))[0]
            t = torch.load(os.path.join(d, f), map_location='cpu')
            resolutions[key] = (t.shape[1], t.shape[2])  # (H, W)
    return resolutions


def render_per_scale(feat_model, viewmat, ref_fx, ref_fy, ref_W, ref_H,
                     scale_resolutions, device):
    """Render each scale at its native resolution with colors_override,
    matching the training rendering in train_multiscale_embedding_v2.py."""
    raw_feat = feat_model._loc_feature  # [N, 224]
    scales = {}

    for label, ch_start, ch_end, res_key in SCALE_DEFS:
        tH, tW = scale_resolutions[res_key]
        # Scale intrinsics to target resolution (same as training)
        sfx = ref_fx * tW / ref_W
        sfy = ref_fy * tH / ref_H
        scx = (ref_W / 2.0 - 0.5) * tW / ref_W  # match training: cx * W/ref_W
        scy = (ref_H / 2.0 - 0.5) * tH / ref_H  # match training: cy * H/ref_H

        # Pre-normalize per-Gaussian features (same as training)
        colors = F.normalize(raw_feat[:, ch_start:ch_end], p=2, dim=-1)

        result = FeatureRenderer.render_features(
            gaussian_model=feat_model, viewmat=viewmat,
            fx=sfx, fy=sfy, cx=scx, cy=scy,
            img_height=tH, img_width=tW,
            norm_feat_before_render=False, norm_feat_after_render=False,
            colors_override=colors,
        )
        fm = F.normalize(result['feature_map'], p=2, dim=0)  # [C, tH, tW]
        scales[label] = pca_colorize(fm)

    return scales


# ── Main rendering ────────────────────────────────────────

def render_scene(scene_cfg, device, out_dir):
    name = scene_cfg['name']
    print(f"\n{'='*60}\n  Rendering {name}\n{'='*60}")
    os.makedirs(out_dir, exist_ok=True)

    # Load geometry model
    geo_ply = scene_cfg['geo_ply']
    geo_model = GaussianFeatureModel(feature_dim=1)
    geo_model.load_ply(geo_ply)
    geo_model = geo_model.to(device)
    print(f"  Geometry: {geo_model.num_gaussians:,} Gaussians")

    # Load feature model
    feat_ply = scene_cfg.get('feat_ply')
    feat_model = None
    scale_resolutions = None
    if feat_ply and os.path.exists(feat_ply):
        feat_model = MultiScaleGaussianModel()
        feat_model.load_ply_with_features(feat_ply)
        feat_model = feat_model.to(device)
        print(f"  Features: {feat_model.num_gaussians:,} Gaussians, dim={feat_model._loc_feature.shape[1]}")
        # Auto-detect native resolutions from compressed features
        comp_dir = scene_cfg.get('compressed_feat_dir')
        if comp_dir and os.path.exists(comp_dir):
            scale_resolutions = detect_scale_resolutions(comp_dir)
            for k, (h, w) in scale_resolutions.items():
                print(f"    {k}: {w}×{h}")

    # Load poses (already as w2c)
    w2c_list = scene_cfg['w2c_list']
    n_poses = len(w2c_list)
    print(f"  Poses: {n_poses} frames")

    frame_indices = scene_cfg.get('frames', [0, n_poses//4, n_poses//2, 3*n_poses//4])
    frame_indices = [min(i, n_poses - 1) for i in frame_indices]

    fx, fy = scene_cfg['fx'], scene_cfg['fy']
    W, H = scene_cfg['width'], scene_cfg['height']
    scale = scene_cfg.get('render_scale', 1.0)
    rW, rH = int(W * scale), int(H * scale)
    rfx, rfy = fx * scale, fy * scale
    rcx, rcy = rW / 2.0, rH / 2.0
    print(f"  Render: {rW}×{rH} (scale={scale})")

    # Storage
    all_rgbs, all_depths = [], []
    scale_names = [sd[0] for sd in SCALE_DEFS]
    all_scale_pcas = {k: [] for k in scale_names}

    for idx in frame_indices:
        w2c = w2c_list[idx]
        viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)

        with torch.no_grad():
            rgb = FeatureRenderer.render_rgb(
                geo_model, viewmat, rfx, rfy, rcx, rcy, rH, rW
            )['rgb'].clamp(0, 1).cpu()

            depth = FeatureRenderer.render_depth(
                geo_model, viewmat, rfx, rfy, rcx, rcy, rH, rW
            ).cpu()

        all_rgbs.append(rgb)
        all_depths.append(depth)

        if feat_model is not None and scale_resolutions is not None:
            with torch.no_grad():
                scale_pcas = render_per_scale(
                    feat_model, viewmat, fx, fy, W, H,
                    scale_resolutions, device
                )
            for k in scale_names:
                all_scale_pcas[k].append(scale_pcas[k])
        else:
            for k in scale_names:
                all_scale_pcas[k].append(None)

        print(f"  Frame {idx}: depth [{depth.min():.2f}, {depth.max():.2f}]")

    # ── Plot summary: 6 rows × N cols ──
    n_frames = len(frame_indices)
    has_feat = feat_model is not None and scale_resolutions is not None
    n_rows = 2 + (len(scale_names) if has_feat else 0)
    row_labels = ['RGB', 'Depth'] + (scale_names if has_feat else [])

    fig, axes = plt.subplots(n_rows, n_frames, figsize=(4.5 * n_frames, 3.5 * n_rows))
    if n_frames == 1:
        axes = axes.reshape(-1, 1)

    for j, idx in enumerate(frame_indices):
        # Row 0: RGB
        axes[0, j].imshow(all_rgbs[j].permute(1, 2, 0).numpy())
        axes[0, j].set_title(f'Frame {idx}', fontsize=10)
        axes[0, j].axis('off')

        # Row 1: Depth
        d = all_depths[j].numpy()
        valid = d > 0
        vmin, vmax = (np.percentile(d[valid], [2, 98]) if valid.any() else (0, 1))
        axes[1, j].imshow(d, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[1, j].axis('off')

        # Rows 2-5: per-scale Feature PCA
        if has_feat:
            for r, sn in enumerate(scale_names):
                pca_img = all_scale_pcas[sn][j]
                if pca_img is not None:
                    axes[2 + r, j].imshow(pca_img)
                axes[2 + r, j].axis('off')

    # Row labels on left
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=10)

    fig.suptitle(f'{name} — RGB / Depth / Multi-scale Feature PCA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    summary_path = os.path.join(out_dir, f'{name}_summary.png')
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {summary_path}")

    # Cleanup
    del geo_model
    if feat_model is not None:
        del feat_model
    torch.cuda.empty_cache()
    return summary_path


if __name__ == '__main__':
    device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES selects physical GPU

    # ── Room_0: traj_w_c.txt (c2w → w2c) ──
    room0_w2c = load_traj_w2c('dataset/room_0/Sequence_1/traj_w_c.txt')

    # ── OldHospital: cameras.json (R_c2w + position → w2c) ──
    oh_w2c, oh_cams = load_cameras_json_w2c(
        'output/2dgs_models/OldHospital/v7_depth/cameras.json'
    )
    print(f"Room_0: {len(room0_w2c)} poses,  OldHospital: {len(oh_w2c)} poses")

    scenes = [
        {
            'name': 'room_0',
            'geo_ply': 'output/2dgs_models/room_0/v8_fixed_poses/point_cloud/iteration_30000/point_cloud.ply',
            'feat_ply': 'output/feature_3dgs/room_0_v1/point_cloud_with_features.ply',
            'compressed_feat_dir': 'output/features_multiscale_compressed/room_0',
            'w2c_list': room0_w2c,
            'fx': 320.0, 'fy': 320.0,
            'width': 640, 'height': 480,
            'render_scale': 1.0,
            'frames': [200, 320, 500, 670],  # PCA-balanced frames (avoid traj edges)
        },
        {
            'name': 'OldHospital',
            'geo_ply': 'output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply',
            'feat_ply': 'output/feature_3dgs/oldhospital_v1/point_cloud_with_features.ply',
            'compressed_feat_dir': 'output/features_multiscale_compressed/OldHospital',
            'w2c_list': oh_w2c,
            'fx': 1663.12, 'fy': 1663.12,
            'width': 1920, 'height': 1080,
            'render_scale': 0.5,
            'frames': [0, 200, 500, 800],
        },
    ]

    for cfg in scenes:
        out_dir = f"output/feature_3dgs/{cfg['name']}_vis"
        try:
            render_scene(cfg, device, out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\nDone!")
