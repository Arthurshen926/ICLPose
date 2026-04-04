#!/usr/bin/env python3
"""
Comprehensive DCFF feature assessment — renders multi-channel visualization grid.

For each scene produces a large grid image:
  Row 1: RGB render | Depth | Alpha
  Row 2: Fine PCA (pred) | Fine PCA (GT) | Fine cosine-sim heatmap
  Row 3: Fine norm heatmap (pred) | Fine norm heatmap (GT) | Norm-diff heatmap
  Row 4: Individual feature channels (6 selected channels) — pred
  Row 5: Individual feature channels (6 selected channels) — GT
  Row 6: Centered PCA (pred) | Centered PCA (GT) | Centered cosine-sim
  Row 7: Coarse PCA (pred) | Coarse PCA (GT) | Coarse cosine-sim
  + Statistics: cosine-sim, L1, norm correlation, effective rank, etc.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/render_feature_assessment.py \
      --scenes Room_0 Stairs OldHospital
"""
import argparse, glob, math, os, sys, yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcff.hybrid_gaussian import HybridGaussianModel
from dcff.hash_grid import SpatialHashGrid
from dcff.deferred_renderer import DeferredCascadedRenderer
from feature_3dgs.train_2dgs_joint_v2 import load_scene_colmap

# ── Scene registry ─────────────────────────────────────────────────

SCENES = {
    'Room_0': {
        'config': 'configs/dcff_room0_v7b.yaml',
        'checkpoint': 'output/dcff_room0_v7b/checkpoints/best.pth',
        'ply': 'output/dcff_room0_v7b/checkpoints/best.ply',
    },
    'Stairs': {
        'config': 'configs/dcff_stairs_v7b.yaml',
        'checkpoint': 'output/dcff_stairs_v7b/checkpoints/best.pth',
        'ply': 'output/dcff_stairs_v7b/checkpoints/best.ply',
    },
    'OldHospital': {
        'config': 'configs/dcff_oldhospital_v7b.yaml',
        'checkpoint': 'output/dcff_oldhospital_v7b/checkpoints/best.pth',
        'ply': 'output/dcff_oldhospital_v7b/checkpoints/best.ply',
    },
}


# ── Utility functions ──────────────────────────────────────────────

def to_np(t):
    if t is None:
        return None
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def to_np_uint8(arr):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        arr = to_np(arr)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def add_label(img_np, text, fontsize=14, bg_alpha=0.7):
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([0, 0, bbox[2] + 8, bbox[3] + 4], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


def _resize(t_chw, h, w):
    return F.interpolate(t_chw.unsqueeze(0).float(), (h, w),
                        mode='bilinear', align_corners=False)[0]


def joint_pca(feats, n_components=3):
    """Joint PCA across list of feature maps [C,H,W]. None preserved."""
    valid = [f for f in feats if f is not None]
    if not valid:
        return [None] * len(feats)
    C, H, W = valid[0].shape
    stacked = torch.cat([f.reshape(C, -1).T.float() for f in valid], 0)
    mean = stacked.mean(0, keepdim=True)
    try:
        _, _, Vh = torch.linalg.svd(stacked - mean, full_matrices=False)
        basis = Vh[:n_components]
    except Exception:
        basis = torch.eye(n_components, C, device=valid[0].device)
    all_proj, all_vp = [], []
    for f in feats:
        if f is None:
            all_proj.append(None); continue
        flat = f.reshape(C, -1).T.float()
        p = (flat - mean) @ basis.T
        all_proj.append(p); all_vp.append(p)
    cat_vp = torch.cat(all_vp, 0)
    lo = cat_vp.min(0).values
    hi = cat_vp.max(0).values
    out = []
    for p in all_proj:
        if p is None:
            out.append(None); continue
        for i in range(n_components):
            p[:, i] = (p[:, i] - lo[i]) / (hi[i] - lo[i] + 1e-8)
        out.append(p.T.reshape(n_components, H, W).clamp(0, 1))
    return out


def norm_heatmap(feat, vmin=None, vmax=None):
    """Feature norm [C,H,W] → [3,H,W] heatmap in [0,1]."""
    norms = feat.norm(dim=0)  # [H,W]
    if vmin is None: vmin = norms.min()
    if vmax is None: vmax = norms.quantile(0.98)
    norms_n = ((norms - vmin) / (vmax - vmin + 1e-6)).clamp(0, 1)
    return colormap_jet(norms_n)


def colormap_jet(val_hw):
    """Jet colormap for [H,W] in [0,1] → [3,H,W]."""
    v = val_hw.detach().cpu().numpy()
    r = np.clip(1.5 - abs(4*v - 3), 0, 1)
    g = np.clip(1.5 - abs(4*v - 2), 0, 1)
    b = np.clip(1.5 - abs(4*v - 1), 0, 1)
    return torch.from_numpy(np.stack([r, g, b], 0)).float().to(val_hw.device)


def cosine_sim_heatmap(pred, gt):
    """Per-pixel cosine → jet heatmap."""
    p = F.normalize(pred, dim=0, eps=1e-6)
    g = F.normalize(gt, dim=0, eps=1e-6)
    sim = (p * g).sum(0).clamp(0, 1)
    return colormap_jet(sim), sim.mean().item()


def channel_vis(feat, ch_indices, vmin=None, vmax=None):
    """Visualize individual channels as heatmaps. Returns list of [3,H,W]."""
    imgs = []
    for ch in ch_indices:
        v = feat[ch]
        if vmin is None: lo = v.min()
        else: lo = vmin
        if vmax is None: hi = v.max()
        else: hi = vmax
        v_n = ((v - lo) / (hi - lo + 1e-6)).clamp(0, 1)
        imgs.append(colormap_jet(v_n))
    return imgs


def depth_vis(depth_1hw, alpha_1hw):
    d = depth_1hw[0]
    valid = d[alpha_1hw[0] > 0.1]
    if len(valid) == 0:
        return torch.zeros(3, *d.shape, device=d.device)
    lo, hi = valid.min(), valid.quantile(0.98)
    d_norm = ((d - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    vis = 1.0 - d_norm
    vis = vis * (alpha_1hw[0] > 0.1).float()
    return colormap_jet(1.0 - vis)


def compute_feature_stats(pred, gt):
    """Compute comprehensive feature statistics."""
    stats = {}
    C, H, W = pred.shape

    # Basic stats
    stats['pred_norm_mean'] = pred.norm(dim=0).mean().item()
    stats['pred_norm_std'] = pred.norm(dim=0).std().item()
    stats['gt_norm_mean'] = gt.norm(dim=0).mean().item()
    stats['gt_norm_std'] = gt.norm(dim=0).std().item()

    # Per-pixel cosine similarity
    cos = (F.normalize(pred, dim=0) * F.normalize(gt, dim=0)).sum(0)
    stats['cosine_mean'] = cos.mean().item()
    stats['cosine_std'] = cos.std().item()
    stats['cosine_min'] = cos.min().item()

    # Raw L1
    stats['l1_mean'] = (pred - gt).abs().mean().item()

    # Effective rank (SVD of pred features)
    flat = pred.reshape(C, -1).T.float()  # [N, C]
    flat_centered = flat - flat.mean(0, keepdim=True)
    try:
        S = torch.linalg.svdvals(flat_centered)
        p = S / S.sum()
        entropy = -(p * (p + 1e-10).log()).sum()
        stats['eff_rank_pred'] = entropy.exp().item()
    except Exception:
        stats['eff_rank_pred'] = -1

    # Same for centered features
    pred_c = pred - pred.mean(dim=(1,2), keepdim=True)
    gt_c = gt - gt.mean(dim=(1,2), keepdim=True)
    cos_c = (F.normalize(pred_c, dim=0) * F.normalize(gt_c, dim=0)).sum(0)
    stats['centered_cosine_mean'] = cos_c.mean().item()

    flat_c = pred_c.reshape(C, -1).T.float()
    flat_c = flat_c - flat_c.mean(0, keepdim=True)
    try:
        S_c = torch.linalg.svdvals(flat_c)
        p_c = S_c / S_c.sum()
        entropy_c = -(p_c * (p_c + 1e-10).log()).sum()
        stats['eff_rank_centered'] = entropy_c.exp().item()
    except Exception:
        stats['eff_rank_centered'] = -1

    # Norm correlation: do the norms match spatially?
    pred_norms = pred.norm(dim=0).flatten()
    gt_norms = gt.norm(dim=0).flatten()
    pn = pred_norms - pred_norms.mean()
    gn = gt_norms - gt_norms.mean()
    corr = (pn * gn).sum() / (pn.norm() * gn.norm() + 1e-8)
    stats['norm_correlation'] = corr.item()

    return stats


# ── Model loading ──────────────────────────────────────────────────

def load_model(config_path, checkpoint_path, ply_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg['model']; hcfg = cfg['hash_grid']; fcfg = cfg.get('fine_decoder', {})

    gaussians = HybridGaussianModel(sh_degree=mcfg['sh_degree'],
                                     latent_dim=mcfg['latent_dim'])
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = mcfg['sh_degree']

    hash_grid = SpatialHashGrid(
        feature_dim=mcfg['feature_dim'], latent_dim=mcfg['latent_dim'],
        input_mode=hcfg.get('input_mode', 'legacy'),
        n_levels=hcfg['n_levels'], n_features_per_level=hcfg['n_features_per_level'],
        log2_hashmap_size=hcfg['log2_hashmap_size'],
        base_resolution=hcfg['base_resolution'], max_resolution=hcfg['max_resolution'],
        mlp_hidden=hcfg.get('mlp_hidden', 256), mlp_layers=hcfg.get('mlp_layers', 4),
        scale_dim=hcfg.get('scale_dim', 2), scale_pe_freqs=hcfg.get('scale_pe_freqs', 4),
        include_raw_scale=hcfg.get('include_raw_scale', True),
    ).to(device)

    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid, latent_dim=mcfg['latent_dim'],
        fine_feature_dim=mcfg['feature_dim'], coarse_feature_dim=mcfg['feature_dim'],
        fine_hidden_dim=fcfg.get('hidden_dim'), fine_num_layers=fcfg.get('num_layers', 3),
        fine_use_viewdirs=fcfg.get('use_viewdirs', False),
        fine_view_degree=fcfg.get('view_degree', 2),
        normalize_features=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'hash_grid_state' in ckpt:
        renderer.hash_grid.load_state_dict(ckpt['hash_grid_state'])
        renderer.fine_decoder.load_state_dict(ckpt['fine_decoder_state'])
    elif 'renderer_state_dict' in ckpt:
        renderer.load_state_dict(ckpt['renderer_state_dict'])
    renderer.eval()
    return cfg, gaussians, renderer


def load_gt_features(feat_dir, cam_name, device):
    """Load GT fine/coarse features for a camera."""
    fine_path = os.path.join(feat_dir, 'fine_geo', f'{cam_name}.pt')
    coarse_path = os.path.join(feat_dir, 'coarse_geo', f'{cam_name}.pt')
    fine = torch.load(fine_path, map_location=device, weights_only=True).float() if os.path.exists(fine_path) else None
    coarse = torch.load(coarse_path, map_location=device, weights_only=True).float() if os.path.exists(coarse_path) else None
    return fine, coarse


def render_one_view(cfg, gaussians, renderer, cam, device, feat_scale=1.0):
    """Render features for one camera view."""
    mcfg = cfg['model']
    longest = cfg.get('training', {}).get('longest_edge', 960)
    ow, oh = cam.width, cam.height
    if ow >= oh:
        rw = longest; rh = int(oh * longest / ow)
    else:
        rh = longest; rw = int(ow * longest / oh)

    # Feature resolution
    feat_dir = cfg['dataset'].get('feature_dir', '')
    fh, fw = rh // 14, rw // 14
    if feat_dir:
        samples = glob.glob(os.path.join(feat_dir, 'fine_geo', '*.pt'))
        if samples:
            t = torch.load(samples[0], map_location='cpu', weights_only=True)
            fh, fw = t.shape[-2], t.shape[-1]

    fh_render = int(fh * feat_scale)
    fw_render = int(fw * feat_scale)

    # Build camera matrices
    import numpy as np_
    W2C = np_.eye(4)
    W2C[:3, :3] = cam.R.T
    W2C[:3, 3] = cam.T
    w2c = torch.tensor(W2C, dtype=torch.float32, device=device)

    fx = rw / (2 * math.tan(cam.FovX * 0.5))
    fy = rh / (2 * math.tan(cam.FovY * 0.5))
    K = torch.tensor([[fx, 0, rw/2.0], [0, fy, rh/2.0], [0, 0, 1]],
                     dtype=torch.float32, device=device)

    with torch.no_grad():
        result = renderer(gaussians, viewmat=w2c, K=K,
                         width=rw, height=rh,
                         render_coarse=True,
                         feature_height=fh_render, feature_width=fw_render)

    out = {
        'rgb': result['rgb'][0],          # [3, rH, rW]
        'depth': result['depth'][0],      # [1, rH, rW]
        'alpha': result['alpha'][0],      # [1, rH, rW]
        'fine': result['fine_features'][0],  # [C, fH, fW]
        'render_size': (rw, rh),
        'feat_size': (fw_render, fh_render),
        'native_feat_size': (fw, fh),
    }
    if 'coarse_features' in result and result['coarse_features'] is not None:
        out['coarse'] = result['coarse_features'][0]
    return out


def make_grid(images, ncols, pad=2, bg_color=0):
    """Stack images into a grid. images: list of [H,W,3] np uint8."""
    images = [img for img in images if img is not None]
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    nrows = (len(images) + ncols - 1) // ncols
    grid = np.full((nrows * (h + pad) - pad, ncols * (w + pad) - pad, 3),
                   int(bg_color * 255), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = i // ncols, i % ncols
        y, x = r * (h + pad), c * (w + pad)
        ih, iw = img.shape[:2]
        grid[y:y+ih, x:x+iw] = img
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='*', default=list(SCENES.keys()))
    parser.add_argument('--num_views', type=int, default=3)
    parser.add_argument('--feat_scale', type=float, default=1.0,
                        help='Feature resolution multiplier for rendering')
    parser.add_argument('--outdir', default='output/feature_assessment')
    args = parser.parse_args()

    device = torch.device('cuda')
    os.makedirs(args.outdir, exist_ok=True)

    for scene_name in args.scenes:
        if scene_name not in SCENES:
            print(f"  Skipping unknown scene: {scene_name}")
            continue

        sinfo = SCENES[scene_name]
        if not os.path.exists(sinfo['checkpoint']):
            print(f"  No checkpoint for {scene_name}")
            continue

        print(f"\n{'='*70}")
        print(f"  Scene: {scene_name}")
        print(f"{'='*70}")

        cfg, gaussians, renderer = load_model(
            sinfo['config'], sinfo['checkpoint'], sinfo['ply'], device)

        train_cams, test_cams, _, _, _ = load_scene_colmap(
            cfg['dataset']['source_dir'], cfg['dataset'].get('images', ''))

        feat_dir = cfg['dataset'].get('feature_dir', '')
        cams = test_cams if len(test_cams) >= args.num_views else train_cams
        indices = np.linspace(0, len(cams)-1, args.num_views, dtype=int)

        all_stats = []

        for vi, idx in enumerate(indices):
            cam = cams[idx]
            cam_name = os.path.splitext(os.path.basename(cam.image_name))[0]
            print(f"\n  View {vi+1}/{args.num_views}: {cam_name}")

            # Render
            out = render_one_view(cfg, gaussians, renderer, cam, device,
                                  feat_scale=args.feat_scale)
            rw, rh = out['render_size']
            fw, fh = out['feat_size']
            nfw, nfh = out['native_feat_size']
            print(f"    Render: {rw}x{rh}, Features: {fw}x{fh} (native {nfw}x{nfh}, scale {args.feat_scale}x)")

            # Load GT
            gt_fine, gt_coarse = None, None
            if feat_dir:
                gt_fine, gt_coarse = load_gt_features(feat_dir, cam_name, device)
                if gt_fine is not None:
                    gt_fine = gt_fine.squeeze(0) if gt_fine.dim() == 4 else gt_fine

            pred_fine = out['fine']
            pred_coarse = out.get('coarse')

            # Match resolution: resize pred to GT size or vice versa
            display_h, display_w = fh, fw
            if gt_fine is not None:
                gt_fh, gt_fw = gt_fine.shape[-2], gt_fine.shape[-1]
                if (gt_fh, gt_fw) != (fh, fw):
                    # Resize pred to match GT for comparison
                    pred_fine_cmp = _resize(pred_fine, gt_fh, gt_fw)
                    display_h, display_w = gt_fh, gt_fw
                else:
                    pred_fine_cmp = pred_fine
            else:
                pred_fine_cmp = pred_fine

            # ── Compute stats ──
            if gt_fine is not None:
                stats = compute_feature_stats(pred_fine_cmp, gt_fine)
                all_stats.append(stats)
                print(f"    cos={stats['cosine_mean']:.4f}, L1={stats['l1_mean']:.4f}")
                print(f"    norm: pred={stats['pred_norm_mean']:.2f}±{stats['pred_norm_std']:.2f}, "
                      f"GT={stats['gt_norm_mean']:.2f}±{stats['gt_norm_std']:.2f}, corr={stats['norm_correlation']:.3f}")
                print(f"    eff_rank: raw={stats['eff_rank_pred']:.1f}, centered={stats['eff_rank_centered']:.1f}")
                print(f"    centered_cos={stats['centered_cosine_mean']:.4f}")

            # ── Build visualization grid ──
            vis_h = 240   # target visualization height
            vis_w = int(vis_h * fw / fh)
            vis_w_rgb = int(vis_h * rw / rh)

            images = []

            # Row 1: RGB | Depth | Alpha
            rgb_vis = _resize(out['rgb'], vis_h, vis_w_rgb)
            depth_vis_img = depth_vis(out['depth'], out['alpha'])
            depth_vis_img = _resize(depth_vis_img, vis_h, vis_w_rgb)
            alpha_vis = _resize(out['alpha'].repeat(3,1,1), vis_h, vis_w_rgb)

            images.append(add_label(to_np_uint8(rgb_vis), f'RGB {rw}x{rh}'))
            images.append(add_label(to_np_uint8(depth_vis_img), 'Depth'))
            images.append(add_label(to_np_uint8(alpha_vis), 'Alpha'))

            # Row 2: Fine PCA — Joint PCA pred vs GT
            pca_list = [pred_fine_cmp]
            if gt_fine is not None:
                pca_list.append(gt_fine)
            pca_results = joint_pca(pca_list)

            pred_pca = _resize(pca_results[0], vis_h, vis_w) if pca_results[0] is not None else None
            gt_pca = _resize(pca_results[1], vis_h, vis_w) if len(pca_results) > 1 and pca_results[1] is not None else None

            images.append(add_label(to_np_uint8(pred_pca), f'Fine PCA (pred) {fw}x{fh}'))
            if gt_pca is not None:
                images.append(add_label(to_np_uint8(gt_pca), 'Fine PCA (GT)'))
            else:
                images.append(np.zeros((vis_h, vis_w, 3), dtype=np.uint8))

            # Cosine sim heatmap
            if gt_fine is not None:
                cos_hm, cos_mean = cosine_sim_heatmap(pred_fine_cmp, gt_fine)
                cos_hm = _resize(cos_hm, vis_h, vis_w)
                images.append(add_label(to_np_uint8(cos_hm), f'Cosine sim={cos_mean:.4f}'))
            else:
                images.append(np.zeros((vis_h, vis_w, 3), dtype=np.uint8))

            # Row 3: Norm heatmaps
            # Compute shared norm range for pred vs GT
            pred_norms = pred_fine_cmp.norm(dim=0)
            if gt_fine is not None:
                gt_norms = gt_fine.norm(dim=0)
                vmin = min(pred_norms.min(), gt_norms.min())
                vmax = max(pred_norms.quantile(0.98), gt_norms.quantile(0.98))
            else:
                vmin, vmax = pred_norms.min(), pred_norms.quantile(0.98)

            pred_norm_vis = _resize(norm_heatmap(pred_fine_cmp, vmin, vmax), vis_h, vis_w)
            images.append(add_label(to_np_uint8(pred_norm_vis),
                         f'Norm (pred) μ={pred_norms.mean():.1f}'))

            if gt_fine is not None:
                gt_norm_vis = _resize(norm_heatmap(gt_fine, vmin, vmax), vis_h, vis_w)
                images.append(add_label(to_np_uint8(gt_norm_vis),
                             f'Norm (GT) μ={gt_norms.mean():.1f}'))

                # Norm difference
                norm_diff = (pred_norms - gt_norms).abs()
                nd_vis = colormap_jet((norm_diff / (vmax - vmin + 1e-6)).clamp(0, 1))
                nd_vis = _resize(nd_vis, vis_h, vis_w)
                images.append(add_label(to_np_uint8(nd_vis),
                             f'|Δnorm| μ={norm_diff.mean():.2f}'))
            else:
                images.append(np.zeros((vis_h, vis_w, 3), dtype=np.uint8))
                images.append(np.zeros((vis_h, vis_w, 3), dtype=np.uint8))

            # Row 4-5: Individual channels (pred and GT)
            # Select channels with highest variance in pred
            ch_var = pred_fine_cmp.var(dim=(1,2))
            top_channels = ch_var.argsort(descending=True)[:6].tolist()

            for prefix, feat in [('Pred', pred_fine_cmp), ('GT', gt_fine)]:
                if feat is None:
                    for _ in range(6):
                        images.append(np.zeros((vis_h, vis_w, 3), dtype=np.uint8))
                    continue
                # Shared range per channel for fair comparison
                for ch in top_channels:
                    ch_pred = pred_fine_cmp[ch]
                    ch_gt = gt_fine[ch] if gt_fine is not None else ch_pred
                    lo = min(ch_pred.min(), ch_gt.min())
                    hi = max(ch_pred.max(), ch_gt.max())
                    v = feat[ch]
                    v_n = ((v - lo) / (hi - lo + 1e-6)).clamp(0, 1)
                    ch_vis = _resize(colormap_jet(v_n), vis_h, vis_w)
                    images.append(add_label(to_np_uint8(ch_vis), f'{prefix} ch{ch}'))

            # Row 6: Centered PCA — after subtracting spatial mean
            pred_c = pred_fine_cmp - pred_fine_cmp.mean(dim=(1,2), keepdim=True)
            gt_c = gt_fine - gt_fine.mean(dim=(1,2), keepdim=True) if gt_fine is not None else None

            c_pca_list = [pred_c]
            if gt_c is not None:
                c_pca_list.append(gt_c)
            c_pca_results = joint_pca(c_pca_list)

            c_pred_pca = _resize(c_pca_results[0], vis_h, vis_w)
            images.append(add_label(to_np_uint8(c_pred_pca), 'Centered PCA (pred)'))

            if len(c_pca_results) > 1 and c_pca_results[1] is not None:
                c_gt_pca = _resize(c_pca_results[1], vis_h, vis_w)
                images.append(add_label(to_np_uint8(c_gt_pca), 'Centered PCA (GT)'))
            else:
                images.append(np.zeros((vis_h, vis_w, 3), dtype=np.uint8))

            if gt_c is not None:
                c_cos_hm, c_cos_mean = cosine_sim_heatmap(pred_c, gt_c)
                c_cos_hm = _resize(c_cos_hm, vis_h, vis_w)
                images.append(add_label(to_np_uint8(c_cos_hm),
                             f'Centered cos={c_cos_mean:.4f}'))
            else:
                images.append(np.zeros((vis_h, vis_w, 3), dtype=np.uint8))

            # Row 7: Coarse features (if available)
            if pred_coarse is not None:
                if gt_coarse is not None:
                    gt_coarse_sq = gt_coarse.squeeze(0) if gt_coarse.dim() == 4 else gt_coarse
                    # Match resolution
                    cfh, cfw = gt_coarse_sq.shape[-2], gt_coarse_sq.shape[-1]
                    pred_coarse_cmp = _resize(pred_coarse, cfh, cfw)
                    coarse_pca = joint_pca([pred_coarse_cmp, gt_coarse_sq])
                    cp_vis = _resize(coarse_pca[0], vis_h, vis_w)
                    cg_vis = _resize(coarse_pca[1], vis_h, vis_w)
                    images.append(add_label(to_np_uint8(cp_vis), 'Coarse PCA (pred)'))
                    images.append(add_label(to_np_uint8(cg_vis), 'Coarse PCA (GT)'))
                    cc_hm, cc_mean = cosine_sim_heatmap(pred_coarse_cmp, gt_coarse_sq)
                    cc_hm = _resize(cc_hm, vis_h, vis_w)
                    images.append(add_label(to_np_uint8(cc_hm),
                                 f'Coarse cos={cc_mean:.4f}'))
                else:
                    cp_pca = joint_pca([pred_coarse])
                    cp_vis = _resize(cp_pca[0], vis_h, vis_w)
                    images.append(add_label(to_np_uint8(cp_vis), 'Coarse PCA (pred)'))
                    images.append(np.zeros((vis_h, vis_w, 3), dtype=np.uint8))
                    images.append(np.zeros((vis_h, vis_w, 3), dtype=np.uint8))

            # Assemble grid: 3 columns per row
            # Pad all images to same size
            max_h = max(img.shape[0] for img in images)
            max_w = max(img.shape[1] for img in images)
            padded = []
            for img in images:
                if img.shape[0] < max_h or img.shape[1] < max_w:
                    p = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                    p[:img.shape[0], :img.shape[1]] = img
                    padded.append(p)
                else:
                    padded.append(img)

            grid = make_grid(padded, ncols=3, pad=4)

            # Save
            view_path = os.path.join(args.outdir, f'{scene_name}_view{vi}_{cam_name}.png')
            Image.fromarray(grid).save(view_path, quality=95)
            print(f"    Saved: {view_path}")

        # Print aggregate stats
        if all_stats:
            print(f"\n  {'─'*50}")
            print(f"  Aggregate ({scene_name}, {len(all_stats)} views):")
            for key in ['cosine_mean', 'centered_cosine_mean', 'l1_mean',
                       'norm_correlation', 'eff_rank_pred', 'eff_rank_centered']:
                vals = [s[key] for s in all_stats]
                print(f"    {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n{'='*70}")
    print(f"All outputs saved to {args.outdir}/")


if __name__ == '__main__':
    main()
