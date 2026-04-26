#!/usr/bin/env python3
"""
2DGS Training Visualization
============================
从已有 checkpoint 加载 2DGS 模型，对随机 test view 渲染，生成可视化对比图。

生成内容:
  1. PSNR 训练曲线 (从 train.log 解析)
  2. GT vs Rendered 对比图 (随机采样 test views)
  3. 渲染深度图 + 法线图 (展示几何质量)
  4. 训练配置摘要

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_2dgs.py
"""

import sys
import os
import re
import math
import json
import argparse
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available, will skip chart generation")

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available")


# ── Parse training logs ──────────────────────────────────────────────────


def parse_training_log(log_path):
    """Extract PSNR metrics and loss from training log."""
    metrics = {
        'iterations': [],
        'psnr': [],
        'toned_psnr': [],
        'masked_psnr': [],
        'loss_iters': [],
        'loss_vals': [],
    }
    if not os.path.exists(log_path):
        return metrics

    with open(log_path, 'r') as f:
        for line in f:
            # Parse PSNR
            m = re.search(r'\[Iter (\d+)\] Test PSNR: ([\d.]+) dB', line)
            if m:
                metrics['iterations'].append(int(m.group(1)))
                metrics['psnr'].append(float(m.group(2)))

            # Parse Toned PSNR
            m2 = re.search(r'Toned PSNR: ([\d.]+) dB', line)
            if m2:
                metrics['toned_psnr'].append(float(m2.group(1)))

            # Parse Masked PSNR
            m3 = re.search(r'Masked PSNR: ([\d.]+) dB', line)
            if m3:
                metrics['masked_psnr'].append(float(m3.group(1)))

            # Parse loss from progress bar: Loss=0.xxxxx
            m4 = re.search(r'Loss=([\d.]+)', line)
            if m4:
                # Extract iteration too
                m_iter = re.search(r'(\d+)/\d+', line)
                if m_iter:
                    it = int(m_iter.group(1))
                    if not metrics['loss_iters'] or it > metrics['loss_iters'][-1]:
                        metrics['loss_iters'].append(it)
                        metrics['loss_vals'].append(float(m4.group(1)))

    return metrics


def plot_psnr_curves(all_metrics, output_path):
    """Plot PSNR curves for all datasets on one figure."""
    if not HAS_MPL:
        return

    n_datasets = len(all_metrics)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    for ax, (name, m) in zip(axes, all_metrics.items()):
        if not m['iterations']:
            ax.set_title(f"{name}\n(no data)")
            continue

        ax.plot(m['iterations'], m['psnr'], 'b-o', label='Raw PSNR', linewidth=2, markersize=6)
        if m['toned_psnr']:
            ax.plot(m['iterations'][:len(m['toned_psnr'])], m['toned_psnr'],
                    'r-s', label='Toned PSNR', linewidth=2, markersize=6)
        if m['masked_psnr']:
            ax.plot(m['iterations'][:len(m['masked_psnr'])], m['masked_psnr'],
                    'g-^', label='Masked PSNR', linewidth=2, markersize=6)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate latest value
        if m['psnr']:
            last_iter = m['iterations'][-1]
            last_psnr = m['psnr'][-1]
            ax.annotate(f'{last_psnr:.2f}', xy=(last_iter, last_psnr),
                       fontsize=9, ha='right', va='bottom', color='blue')
        if m['toned_psnr']:
            last_tp = m['toned_psnr'][-1]
            ax.annotate(f'{last_tp:.2f}', xy=(m['iterations'][len(m['toned_psnr'])-1], last_tp),
                       fontsize=9, ha='right', va='bottom', color='red')

    plt.suptitle('2DGS Training PSNR Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved PSNR curves → {output_path}")


def plot_loss_curves(all_metrics, output_path):
    """Plot loss curves."""
    if not HAS_MPL:
        return

    n_datasets = sum(1 for m in all_metrics.values() if m['loss_iters'])
    if n_datasets == 0:
        return

    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 4))
    if n_datasets == 1:
        axes = [axes]

    idx = 0
    for name, m in all_metrics.items():
        if not m['loss_iters']:
            continue
        ax = axes[idx]
        # Subsample for readability
        step = max(1, len(m['loss_iters']) // 200)
        iters = m['loss_iters'][::step]
        losses = m['loss_vals'][::step]
        ax.plot(iters, losses, 'b-', alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name} Loss')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        idx += 1

    plt.suptitle('2DGS Training Loss', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved loss curves → {output_path}")


# ── Checkpoint rendering ─────────────────────────────────────────────────


def render_comparison(dataset_name, source_dir, model_dir, output_dir,
                      n_views=6, longest_edge=0):
    """Load checkpoint and render GT vs Rendered comparisons."""
    from feature_3dgs.train_2dgs_geometry import (
        GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor,
        WildGaussiansAppearance
    )

    # Find latest checkpoint
    pc_dir = os.path.join(model_dir, "point_cloud")
    if not os.path.exists(pc_dir):
        print(f"  [{dataset_name}] No checkpoint directory found, skipping render")
        return

    ckpt_dirs = sorted([d for d in os.listdir(pc_dir) if d.startswith("iteration_")],
                       key=lambda x: int(x.split("_")[1]))
    if not ckpt_dirs:
        print(f"  [{dataset_name}] No checkpoints found")
        return

    latest_ckpt = ckpt_dirs[-1]
    ckpt_iter = int(latest_ckpt.split("_")[1])
    ply_path = os.path.join(pc_dir, latest_ckpt, "point_cloud.ply")
    appearance_path = os.path.join(pc_dir, latest_ckpt, "appearance_net.pth")

    print(f"\n  [{dataset_name}] Loading checkpoint @ iter {ckpt_iter}")
    print(f"    PLY: {ply_path}")

    # Load model
    gaussians = GaussianModel2DGS(sh_degree=3)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = 3

    # Load appearance network if exists
    appearance_net = None
    use_wg = False
    if os.path.exists(appearance_path):
        ckpt_data = torch.load(appearance_path, map_location='cuda')
        if ckpt_data.get('type') == 'wildgaussians':
            # Infer hidden_dim and n_hidden from state_dict
            sd = ckpt_data['state_dict']
            hidden_dim = sd['mlp.0.bias'].shape[0]
            # Count MLP layers: mlp.0, mlp.2, mlp.4, ... (ReLU at odd indices)
            mlp_weight_keys = sorted([k for k in sd if k.startswith('mlp.') and k.endswith('.weight')])
            n_hidden = len(mlp_weight_keys) - 1  # exclude output layer

            appearance_net = WildGaussiansAppearance(
                n_images=ckpt_data['n_images'],
                n_gaussians=ckpt_data['n_gaussians'],
                image_embed_dim=ckpt_data['image_embed_dim'],
                gaussian_embed_dim=ckpt_data['gaussian_embed_dim'],
                hidden_dim=hidden_dim,
                n_hidden=n_hidden,
            )
            appearance_net.load_state_dict(sd)
            appearance_net.eval().cuda()
            use_wg = True
            print(f"    Loaded WildGaussians appearance (hidden={hidden_dim}, layers={n_hidden})")

    # Monkey-patch args for load_scene (it references global args.max_init_points)
    import argparse
    import feature_3dgs.train_2dgs_geometry as _t2dgs_mod
    if not hasattr(_t2dgs_mod, 'args') or _t2dgs_mod.args is None:
        _t2dgs_mod.args = argparse.Namespace(max_init_points=100000)

    # Load scene
    _, test_cams, _, _, _ = load_scene(source_dir)
    if not test_cams:
        print(f"    No test cameras, using first {n_views} training cameras")
        test_cams, _, _, _, _ = load_scene(source_dir, eval_split=False)

    # Random sample views
    rng = np.random.RandomState(42)
    n_sample = min(n_views, len(test_cams))
    indices = rng.choice(len(test_cams), n_sample, replace=False)
    sample_cams = [test_cams[i] for i in sorted(indices)]

    bg_color = torch.rand(3, device="cuda") if False else torch.zeros(3, device="cuda")

    # Render each view
    renders = []
    gts = []
    depths = []
    normals = []
    view_names = []

    for cam in sample_cams:
        with torch.no_grad():
            pkg = render_2dgs(gaussians, cam, bg_color, longest_edge=longest_edge)
            rendered = pkg["render"].clamp(0, 1)
            depth = pkg["depth"]
            normal = pkg.get("rend_normal", None)

            gt = load_image_tensor(cam)
            rh, rw = pkg["height"], pkg["width"]
            gt = F.interpolate(gt.unsqueeze(0), size=(rh, rw), mode="bilinear",
                              align_corners=False).squeeze(0)

            # WildGaussians toned render
            if use_wg and appearance_net is not None:
                base_colors = appearance_net.get_base_colors(gaussians, cam)
                N_g = base_colors.shape[0]
                gauss_emb = appearance_net.gaussian_embedding.detach()
                mean_emb = appearance_net.image_embedding.weight.mean(dim=0).detach()
                emb_expanded = mean_emb.unsqueeze(0).expand(N_g, -1)
                mlp_in = torch.cat([emb_expanded, gauss_emb, base_colors.detach()], dim=1)
                out = appearance_net.mlp(mlp_in)
                s = appearance_net.output_scale * out[:, :3] + 1.0
                b = appearance_net.output_scale * out[:, 3:]
                toned_colors = (s * base_colors.detach() + b).clamp(0, 1)
                toned_pkg = render_2dgs(gaussians, cam, bg_color,
                                        longest_edge=longest_edge,
                                        override_colors=toned_colors)
                toned = toned_pkg["render"].clamp(0, 1)
                # Use toned as rendered (better visual quality)
                rendered = toned

        renders.append(rendered.cpu())
        gts.append(gt.cpu())
        depths.append(depth.cpu() if depth is not None else None)
        normals.append(normal.cpu() if normal is not None else None)
        view_names.append(os.path.basename(cam.image_name))

    # Create comparison grid
    os.makedirs(output_dir, exist_ok=True)
    _save_comparison_grid(dataset_name, ckpt_iter, gts, renders, depths, normals,
                          view_names, output_dir, gaussians.num_points)
    print(f"    Rendered {n_sample} views from {dataset_name}")

    # Clean up GPU memory
    del gaussians, appearance_net
    torch.cuda.empty_cache()


def _depth_to_colormap(depth_tensor, valid_mask=None):
    """Convert depth tensor to colored visualization [3, H, W] float."""
    d = depth_tensor[0] if depth_tensor.dim() == 3 else depth_tensor  # [H, W]
    d = d.float()
    if valid_mask is not None:
        valid = valid_mask.float()
    else:
        valid = (d > 0).float()

    d_valid = d[valid > 0.5]
    if d_valid.numel() == 0:
        return torch.zeros(3, d.shape[0], d.shape[1])

    vmin = d_valid.quantile(0.02)
    vmax = d_valid.quantile(0.98)
    d_norm = ((d - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)

    # Turbo-like colormap via simple interpolation
    r = torch.clamp(1.5 - torch.abs(d_norm * 4 - 3), 0, 1)
    g = torch.clamp(1.5 - torch.abs(d_norm * 4 - 2), 0, 1)
    b = torch.clamp(1.5 - torch.abs(d_norm * 4 - 1), 0, 1)

    rgb = torch.stack([r, g, b], dim=0) * valid.unsqueeze(0)
    return rgb


def _normal_to_vis(normal_tensor):
    """Convert normal map to visualization [3, H, W]."""
    if normal_tensor is None:
        return None
    # Normals are in [-1, 1], map to [0, 1]
    if normal_tensor.dim() == 4:
        normal_tensor = normal_tensor[0]  # remove batch dim
    n = normal_tensor[:3]  # [3, H, W]
    return (n * 0.5 + 0.5).clamp(0, 1)


def _save_comparison_grid(dataset_name, ckpt_iter, gts, renders, depths, normals,
                           view_names, output_dir, n_gaussians):
    """Save comparison images."""
    if not HAS_MPL or not HAS_PIL:
        # Fallback: save individual images
        for i, (gt, rend) in enumerate(zip(gts, renders)):
            gt_img = PILImage.fromarray((gt.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            rend_img = PILImage.fromarray((rend.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            gt_img.save(os.path.join(output_dir, f"{dataset_name}_view{i}_gt.png"))
            rend_img.save(os.path.join(output_dir, f"{dataset_name}_view{i}_render.png"))
        return

    n_views = len(gts)
    has_depth = any(d is not None for d in depths)
    has_normal = any(n is not None for n in normals)

    n_rows = 2  # GT + Render
    if has_depth:
        n_rows += 1
    if has_normal:
        n_rows += 1

    fig, axes = plt.subplots(n_rows, n_views, figsize=(4 * n_views, 4 * n_rows))
    if n_views == 1:
        axes = axes[:, np.newaxis]

    row_labels = ['Ground Truth', f'Render @{ckpt_iter}']
    if has_depth:
        row_labels.append('Depth')
    if has_normal:
        row_labels.append('Normal')

    for i in range(n_views):
        # GT
        axes[0, i].imshow(gts[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(view_names[i], fontsize=8)
        axes[0, i].axis('off')

        # Render
        axes[1, i].imshow(renders[i].permute(1, 2, 0).numpy())
        # Per-pixel error
        psnr_per = -10 * torch.log10(
            ((gts[i] - renders[i]) ** 2).mean() + 1e-8).item()
        axes[1, i].set_title(f'PSNR: {psnr_per:.1f}dB', fontsize=8)
        axes[1, i].axis('off')

        row_idx = 2

        # Depth
        if has_depth and depths[i] is not None:
            depth_vis = _depth_to_colormap(depths[i])
            axes[row_idx, i].imshow(depth_vis.permute(1, 2, 0).numpy())
            axes[row_idx, i].axis('off')
            row_idx += 1

        # Normal
        if has_normal and normals[i] is not None:
            normal_vis = _normal_to_vis(normals[i])
            if normal_vis is not None:
                axes[row_idx, i].imshow(normal_vis.permute(1, 2, 0).numpy())
            axes[row_idx, i].axis('off')

    # Row labels
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=11, fontweight='bold')

    plt.suptitle(f'{dataset_name} — 2DGS Render @iter {ckpt_iter} ({n_gaussians:,} Gaussians)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{dataset_name}_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved → {out_path}")

    # Also save error map
    fig2, axes2 = plt.subplots(1, n_views, figsize=(4 * n_views, 4))
    if n_views == 1:
        axes2 = [axes2]
    for i in range(n_views):
        err = ((gts[i] - renders[i]) ** 2).mean(dim=0).sqrt()  # [H, W] RMSE
        im = axes2[i].imshow(err.numpy(), cmap='hot', vmin=0, vmax=0.3)
        axes2[i].set_title(view_names[i], fontsize=8)
        axes2[i].axis('off')
    plt.colorbar(im, ax=axes2, fraction=0.02)
    plt.suptitle(f'{dataset_name} — RMSE Error Map @iter {ckpt_iter}', fontsize=13)
    plt.tight_layout()
    err_path = os.path.join(output_dir, f"{dataset_name}_error_map.png")
    plt.savefig(err_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved → {err_path}")


# ── Summary report ───────────────────────────────────────────────────────


def generate_summary(all_metrics, datasets_info, output_path):
    """Generate a text summary of training status."""
    lines = [
        "=" * 70,
        "  2DGS Training Status Report",
        "=" * 70,
        "",
    ]

    for name, info in datasets_info.items():
        m = all_metrics.get(name, {})
        lines.append(f"▸ {name}")
        lines.append(f"  Model dir: {info['model_dir']}")
        lines.append(f"  Source:    {info['source_dir']}")

        status = info.get('status', 'unknown')
        lines.append(f"  Status:    {status}")

        if m.get('psnr'):
            latest_psnr = m['psnr'][-1]
            best_psnr = max(m['psnr'])
            best_iter = m['iterations'][m['psnr'].index(best_psnr)]
            lines.append(f"  Latest PSNR: {latest_psnr:.2f} dB @ iter {m['iterations'][-1]}")
            lines.append(f"  Best PSNR:   {best_psnr:.2f} dB @ iter {best_iter}")
        if m.get('toned_psnr'):
            latest_tp = m['toned_psnr'][-1]
            best_tp = max(m['toned_psnr'])
            lines.append(f"  Toned PSNR:  {latest_tp:.2f} dB (best: {best_tp:.2f})")

        lines.append(f"  Config: {info.get('config', 'N/A')}")
        lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\n{report}")
    return report


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Visualize 2DGS training results")
    parser.add_argument("--output_dir", default="output/2dgs_visualization",
                       help="Output directory for visualizations")
    parser.add_argument("--n_views", type=int, default=6,
                       help="Number of test views to render per dataset")
    parser.add_argument("--skip_render", action="store_true",
                       help="Skip rendering (only generate charts from logs)")
    parser.add_argument("--gpu", type=int, default=1,
                       help="GPU index (default: 1 to avoid interfering with training)")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Dataset configurations ──
    datasets = {
        'OldHospital': {
            'source_dir': 'dataset/OldHospital',
            'model_dir': 'output/2dgs_models/OldHospital/v3_retrain38g',
            'config': 'WildGaussians, batch=16, 200K max, no normal/dist reg, opacity_reset=30001',
            'longest_edge': 0,
        },
        'stairs': {
            'source_dir': 'dataset/stairs',
            'model_dir': 'output/2dgs_models/stairs/v1_baseline',
            'config': 'Standard 2DGS, batch=32, 500K max, normal=0.05, dist=0.01, opacity_reset=3000',
            'longest_edge': 0,
        },
        'room_0': {
            'source_dir': 'dataset/room_0',
            'model_dir': 'output/2dgs_models/room_0/v1_baseline',
            'config': 'DEAD — diverged from 153K mesh-sampled points, no SfM points',
            'longest_edge': 0,
        },
    }

    print("=" * 70)
    print("  2DGS Training Visualization")
    print("=" * 70)

    # 1. Parse all training logs
    print("\n[1/4] Parsing training logs...")
    all_metrics = {}
    datasets_info = {}
    for name, cfg in datasets.items():
        log_path = os.path.join(cfg['model_dir'], 'train.log')
        m = parse_training_log(log_path)
        all_metrics[name] = m
        datasets_info[name] = {
            **cfg,
            'status': 'running' if m['psnr'] else 'dead/no data',
        }
        if m['psnr']:
            print(f"  {name}: {len(m['psnr'])} eval points, latest PSNR={m['psnr'][-1]:.2f} dB")
        else:
            print(f"  {name}: no PSNR data")

    # 2. Plot PSNR curves
    print("\n[2/4] Generating PSNR charts...")
    plot_psnr_curves(all_metrics, os.path.join(output_dir, "psnr_curves.png"))
    plot_loss_curves(all_metrics, os.path.join(output_dir, "loss_curves.png"))

    # 3. Render comparisons
    if not args.skip_render:
        print("\n[3/4] Rendering comparison views...")
        for name, cfg in datasets.items():
            ply_dir = os.path.join(cfg['model_dir'], 'point_cloud')
            if os.path.exists(ply_dir) and os.listdir(ply_dir):
                try:
                    render_comparison(
                        name, cfg['source_dir'], cfg['model_dir'],
                        output_dir, n_views=args.n_views,
                        longest_edge=cfg['longest_edge'])
                except Exception as e:
                    print(f"    [{name}] Render failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"    [{name}] No checkpoints, skipping render")
    else:
        print("\n[3/4] Skipping render (--skip_render)")

    # 4. Generate summary
    print("\n[4/4] Generating summary report...")
    generate_summary(all_metrics, datasets_info,
                    os.path.join(output_dir, "training_report.txt"))

    print(f"\nAll outputs saved to: {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
