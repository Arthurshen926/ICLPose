#!/usr/bin/env python3
"""
Compare feature upsampling strategies for pose estimation:
  1. Raw DINOv2 patch features (40×70 native resolution)
  2. Direct bilinear upsample of DINOv2 (2×, 4×)
  3. FlowFeat DPT stages (refinenet, no_norm, final)
  4. Depth Anything 3 DPT intermediate features

Outputs:
  - PCA visualization (3-channel RGB) for each method at each scale
  - View consistency metrics (cosine sim between nearby frames)
  - Spatial discriminativeness metrics
"""

import sys, os, math, argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# FlowFeat
FLOWFEAT_ROOT = os.environ.get("FLOWFEAT_ROOT", "/root/flowfeat")
if FLOWFEAT_ROOT not in sys.path:
    sys.path.insert(0, FLOWFEAT_ROOT)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_and_preprocess(path, resize_factor=0.5, patch_size=14, device='cuda'):
    """Load image, resize, normalize, pad to patch-aligned."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    h_new, w_new = int(h * resize_factor), int(w * resize_factor)
    img = img.resize((w_new, h_new), Image.BILINEAR)

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    x = normalize(T.functional.to_tensor(img))

    ps = patch_size
    h_pad = math.ceil(h_new / ps) * ps
    w_pad = math.ceil(w_new / ps) * ps
    gh, gw = h_pad // ps, w_pad // ps
    if gh % 2 != 0: h_pad += ps
    if gw % 2 != 0: w_pad += ps
    x = F.pad(x, (0, w_pad - w_new, 0, h_pad - h_new), mode='reflect')
    return x.unsqueeze(0).to(device), h_pad, w_pad, img


# ─── Method 1 & 2: Raw DINOv2 ───
def extract_dinov2_raw(model, x, h_pad, w_pad, patch_size=14):
    """Extract raw DINOv2 patch features from multiple layers."""
    enc = model.encoder
    enc(x)

    results = {}
    # Layer 4 (deepest) = most semantic
    raw = enc.layer_4[:, 1:]  # skip CLS
    raw = enc.norm(raw)
    gh, gw = h_pad // patch_size, w_pad // patch_size
    raw = raw.movedim(1, -1).view(1, -1, gh, gw)  # (1, 768, gh, gw)

    results['dino_native'] = raw.squeeze(0)  # 768d @ 40×70
    # Bilinear upsample to 2× and 4×
    results['dino_2x'] = F.interpolate(raw, scale_factor=2, mode='bilinear',
                                        align_corners=False).squeeze(0)  # 768d @ 80×140
    results['dino_4x'] = F.interpolate(raw, scale_factor=4, mode='bilinear',
                                        align_corners=False).squeeze(0)  # 768d @ 160×280

    # Also extract layer_1 (shallowest, more local/edge info)
    raw1 = enc.layer_1[:, 1:]
    raw1 = raw1.movedim(1, -1).view(1, -1, gh, gw)
    results['dino_layer1_native'] = raw1.squeeze(0)

    return results


# ─── Method 3 & 4: FlowFeat DPT ───
def extract_flowfeat_stages(model, x, h_pad, w_pad, patch_size=14):
    """Extract features at different DPT decoder stages."""
    enc = model.encoder
    dec = model.decoder
    enc(x)

    results = {}

    # Postprocess
    pp1 = dec.act_postprocess1(enc.layer_1, (h_pad, w_pad))
    pp2 = dec.act_postprocess2(enc.layer_2, (h_pad, w_pad))
    pp3 = dec.act_postprocess3(enc.layer_3, (h_pad, w_pad))
    pp4 = dec.act_postprocess4(enc.layer_4, (h_pad, w_pad))

    # Scratch projection
    l1_rn = dec.scratch.layer1_rn(pp1)
    l2_rn = dec.scratch.layer2_rn(pp2)
    l3_rn = dec.scratch.layer3_rn(pp3)
    l4_rn = dec.scratch.layer4_rn(pp4)

    # RefineNet cascade
    path_4 = dec.scratch.refinenet4(l4_rn)
    path_3 = dec.scratch.refinenet3(path_4, l3_rn)
    path_2 = dec.scratch.refinenet2(path_3, l2_rn)
    path_1 = dec.scratch.refinenet1(path_2, l1_rn)

    # Output convs
    out0 = dec.scratch.output_conv0(path_1)
    out_up = F.interpolate(out0, (h_pad, w_pad), mode='bilinear', align_corners=False)
    out_no_norm = dec.scratch.output_conv1(out_up)
    out_with_norm = dec.norm(out_no_norm)

    results['ff_refinenet'] = path_1.squeeze(0)   # 128d @ high-res (4× patch grid)
    results['ff_no_norm'] = out_no_norm.squeeze(0)  # 128d @ input res
    results['ff_final'] = out_with_norm.squeeze(0)   # 128d @ input res (with LayerNorm)

    return results


# ─── Method 5: Depth Anything 3 DPT ───
def try_load_da3(device='cuda'):
    """Try to load Depth Anything 3 Base model."""
    try:
        from safetensors.torch import load_file
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY

        config = load_config(MODEL_REGISTRY['da3-base'])
        model = create_object(config)

        ckpt_path = '/root/.cache/da3-base/model.safetensors'
        if os.path.exists(ckpt_path):
            weights = load_file(ckpt_path)
            stripped = {k.replace('model.', '', 1): v for k, v in weights.items()}
            missing, _ = model.load_state_dict(stripped, strict=False)
            print(f'[DA3] Loaded pretrained weights (missing={len(missing)} aux keys)')
        else:
            print('[DA3] Warning: no pretrained weights, using random init')

        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    except Exception as e:
        print(f'[DA3] Failed to load: {e}')
        import traceback; traceback.print_exc()
        return None


def extract_da3_stages(model, x, h_pad, w_pad):
    """Extract intermediate DPT features from Depth Anything 3.

    DA3-Base architecture:
      backbone: DinoV2 ViT-B/14 (768d), out_layers=[5,7,9,11], cat_token=True → 1536d
      head: DualDPT(dim_in=1536, features=128, out_channels=[96,192,384,768])
        norm: LayerNorm(1536)
        projects[0..3]: Conv2d(1536→[96,192,384,768])
        resize_layers[0..3]: ConvTranspose(s4), ConvTranspose(s2), Identity, Conv(s2)
        scratch.layer{1..4}_rn: Conv2d(→128)
        scratch.refinenet{4→1}: FeatureFusionBlock(128d) -- main branch
        scratch.output_conv1: Conv2d(128→64)
        scratch.output_conv2: Conv2d(64→32)+ReLU+Conv2d(32→2)  -- depth+conf
    """
    results = {}
    try:
        enc = model.backbone
        head = model.head
        patch_h, patch_w = h_pad // 14, w_pad // 14

        # DA3 expects (B, S, 3, H, W) where S = number of views
        # x is (1, 3, H, W) → reshape to (1, 1, 3, H, W)
        x_da3 = x.unsqueeze(1) if x.dim() == 4 else x

        # Forward backbone: returns (feats_list, aux_feats)
        # Each feat is tuple(outputs, camera_tokens)
        # outputs shape after get_intermediate_layers: (B, S, N_patch, C)
        # where C=1536 for vitb with cat_token=True
        feats, _ = enc(x_da3, export_feat_layers=[])

        # feats is tuple of 4 items: ((output_tensor, cam_token), ...)
        # output_tensor: (B, S, N_patch, C) = (1, 1, patch_h*patch_w, 1536)
        B_S = 1
        C = feats[0][0].shape[-1]  # 1536

        # Flatten B*S dimension
        flat_feats = [feat[0].reshape(B_S, -1, C) for feat in feats]

        # Process through DualDPT head manually
        resized = []
        for stage_idx in range(4):
            tok = flat_feats[stage_idx]  # (1, N_patch, 1536)
            tok = head.norm(tok)
            tok = tok.permute(0, 2, 1).contiguous().reshape(B_S, C, patch_h, patch_w)
            tok = head.projects[stage_idx](tok)
            tok = head.resize_layers[stage_idx](tok)
            resized.append(tok)

        # Main fusion chain
        l1_rn = head.scratch.layer1_rn(resized[0])
        l2_rn = head.scratch.layer2_rn(resized[1])
        l3_rn = head.scratch.layer3_rn(resized[2])
        l4_rn = head.scratch.layer4_rn(resized[3])

        path_4 = head.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        path_3 = head.scratch.refinenet3(path_4, l3_rn, size=l2_rn.shape[2:])
        path_2 = head.scratch.refinenet2(path_3, l2_rn, size=l1_rn.shape[2:])
        path_1 = head.scratch.refinenet1(path_2, l1_rn)

        results['da3_refinenet'] = path_1.squeeze(0)  # 128d

        # output_conv1: 128→64
        out1 = head.scratch.output_conv1(path_1)
        out1_up = F.interpolate(out1, (patch_h * 14, patch_w * 14),
                                mode='bilinear', align_corners=True)
        results['da3_64d'] = out1_up.squeeze(0)  # 64d @ input res

        # output_conv2 partial: first layer only (Conv2d 64→32 + ReLU)
        out2_partial = head.scratch.output_conv2[:2](out1_up)  # 32d
        results['da3_32d'] = out2_partial.squeeze(0)  # 32d @ input res

    except Exception as e:
        print(f'[DA3] Extraction failed: {e}')
        import traceback; traceback.print_exc()

    return results


def pca_colorize(feat, n_components=3, pca_model=None):
    """PCA a (C, H, W) feature to (3, H, W) RGB, return as uint8 numpy HWC."""
    C, H, W = feat.shape
    flat = feat.reshape(C, -1).T.numpy()  # (N, C)

    if pca_model is None:
        pca = PCA(n_components=n_components, random_state=42)
        rgb = pca.fit_transform(flat)  # (N, 3)
    else:
        rgb = pca_model.transform(flat)[:, :3]

    # Normalize each channel to [0, 1] independently
    for c in range(3):
        lo, hi = np.percentile(rgb[:, c], [2, 98])
        rgb[:, c] = np.clip((rgb[:, c] - lo) / (hi - lo + 1e-8), 0, 1)

    return (rgb.reshape(H, W, 3) * 255).astype(np.uint8)


def compute_metrics(feat):
    """Compute spatial discriminativeness metrics for a feature map."""
    C, H, W = feat.shape
    pixel_norm = feat.norm(dim=0)
    cv = (pixel_norm.std() / (pixel_norm.mean() + 1e-8)).item()

    ch_energy = (feat ** 2).sum(dim=(1, 2))
    total = ch_energy.sum()
    p = ch_energy / (total + 1e-8)
    entropy = -(p * (p + 1e-10).log()).sum()
    eff_dim = entropy.exp().item()

    ch_std = feat.std(dim=(1, 2))
    ch_ratio = (ch_std.max() / (ch_std.min() + 1e-8)).item()

    return {
        'spatial_cv': cv,
        'eff_dim': eff_dim,
        'total_dim': C,
        'pixel_norm_mean': pixel_norm.mean().item(),
        'pixel_norm_std': pixel_norm.std().item(),
        'ch_std_ratio': ch_ratio,
    }


def compute_view_consistency(feats_a, feats_b, target_size=(20, 35)):
    """Compute mean cosine similarity between two feature maps at target_size."""
    fa = F.adaptive_avg_pool2d(feats_a.unsqueeze(0), target_size).squeeze(0)
    fb = F.adaptive_avg_pool2d(feats_b.unsqueeze(0), target_size).squeeze(0)
    fa_n = F.normalize(fa.reshape(fa.shape[0], -1), dim=0)
    fb_n = F.normalize(fb.reshape(fb.shape[0], -1), dim=0)
    return (fa_n * fb_n).sum(0).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--output', type=str, default='output/upsample_comparison')
    parser.add_argument('--seq_dir', type=str, default='dataset/OldHospital/seq1')
    parser.add_argument('--frames', type=str, default='1,6,50',
                        help='Comma-separated frame indices to analyze')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    os.makedirs(args.output, exist_ok=True)

    frame_indices = [int(x) for x in args.frames.split(',')]
    frame_paths = [os.path.join(args.seq_dir, f'frame{i:05d}.png') for i in frame_indices]
    for p in frame_paths:
        assert os.path.exists(p), f"Frame not found: {p}"

    print("=" * 60)
    print("  Feature Upsampling Comparison")
    print("=" * 60)

    # ─── Load FlowFeat model ───
    print("\n[1] Loading FlowFeat model...")
    from hubconf import flowfeat
    ff_model = flowfeat('dinov2_vitb14_yt', pretrained=True, map_location=device)
    ff_model = ff_model.to(device).eval()
    for p in ff_model.parameters():
        p.requires_grad_(False)

    # ─── Load DA3 model (optional) ───
    print("\n[2] Loading Depth Anything 3...")
    da3_model = try_load_da3(device=device)
    if da3_model is None:
        print("  DA3 not available, skipping")

    # ─── Extract features for all frames ───
    all_feats = {}  # {frame_idx: {method_name: (C, H, W) tensor}}

    for fi, fp in zip(frame_indices, frame_paths):
        print(f"\n[Frame {fi}] {fp}")
        x, h_pad, w_pad, orig_img = load_and_preprocess(fp, device=device)

        feats = {}

        # Save original image
        feats['_orig_img'] = orig_img

        with torch.no_grad():
            # Methods 1-2: Raw DINOv2
            dino = extract_dinov2_raw(ff_model, x, h_pad, w_pad)
            for k, v in dino.items():
                feats[k] = v.cpu()

            # Methods 3-4: FlowFeat DPT stages
            ff = extract_flowfeat_stages(ff_model, x, h_pad, w_pad)
            for k, v in ff.items():
                feats[k] = v.cpu()

            # Method 5: DA3 (if available)
            if da3_model is not None:
                da3 = extract_da3_stages(da3_model, x, h_pad, w_pad)
                for k, v in da3.items():
                    feats[k] = v.cpu()

        all_feats[fi] = feats

        # Print metrics
        print(f"  {'Method':<25} {'Dim':>5} {'Size':>10} {'SpaCV':>8} {'EffDim':>8} {'PxNorm':>12}")
        print("  " + "-" * 75)
        for name, feat in feats.items():
            if name.startswith('_'):
                continue
            m = compute_metrics(feat)
            C, H, W = feat.shape
            print(f"  {name:<25} {C:>5} {H:>4}×{W:<4} {m['spatial_cv']:>8.4f} "
                  f"{m['eff_dim']:>7.1f} {m['pixel_norm_mean']:>6.2f}±{m['pixel_norm_std']:.2f}")

    # ─── View Consistency ───
    if len(frame_indices) >= 2:
        print(f"\n{'='*60}")
        print(f"  View Consistency (cosine sim at 20×35)")
        print(f"{'='*60}")

        ref_idx = frame_indices[0]
        method_names = [k for k in all_feats[ref_idx] if not k.startswith('_')]

        print(f"\n  {'Method':<25}", end='')
        for fi in frame_indices[1:]:
            print(f"  f{ref_idx}-f{fi:>3}", end='')
        print()
        print("  " + "-" * (25 + 10 * (len(frame_indices) - 1)))

        for name in method_names:
            print(f"  {name:<25}", end='')
            for fi in frame_indices[1:]:
                cos = compute_view_consistency(
                    all_feats[ref_idx][name], all_feats[fi][name])
                print(f"  {cos:>8.4f}", end='')
            print()

    # ─── PCA Visualization ───
    print(f"\n[Generating PCA visualizations...]")

    # Define display groups for visualization
    display_methods = [
        ('dino_native', 'DINOv2 native\n(768d@40×70)'),
        ('dino_2x', 'DINOv2 bilinear 2×\n(768d@80×140)'),
        ('dino_4x', 'DINOv2 bilinear 4×\n(768d@160×280)'),
        ('ff_refinenet', 'FlowFeat RefineNet\n(128d@high-res)'),
        ('ff_no_norm', 'FlowFeat DPT\n(128d, no LayerNorm)'),
        ('ff_final', 'FlowFeat DPT FINAL\n(128d, +LayerNorm)'),
    ]

    # Add DA3 if available
    if da3_model is not None:
        display_methods.append(('da3_refinenet', 'DA3 RefineNet\n(128d)'))
        display_methods.append(('da3_64d', 'DA3 64d\n(after conv1)'))
        display_methods.append(('da3_32d', 'DA3 32d\n(before depth)'))

    # Filter to methods that exist
    display_methods = [(k, label) for k, label in display_methods
                       if k in all_feats[frame_indices[0]]]

    n_methods = len(display_methods)
    n_frames = len(frame_indices)

    fig, axes = plt.subplots(n_frames + 1, n_methods + 1, figsize=(4 * (n_methods + 1), 4 * (n_frames + 1)))
    if n_frames == 1:
        axes = axes[np.newaxis, :]

    # Row 0: header with first frame original image + method names
    for j in range(n_methods + 1):
        axes[0, j].axis('off')

    # Original images in first column
    axes[0, 0].set_title("Original Image", fontsize=11, fontweight='bold')

    for i, fi in enumerate(frame_indices):
        row = i  # row 0 = first frame
        orig = all_feats[fi]['_orig_img']
        axes[row, 0].imshow(orig)
        axes[row, 0].set_title(f"Frame {fi}", fontsize=10)
        axes[row, 0].axis('off')

    # PCA fits on first frame, then apply to all
    # For fair comparison, fit separate PCA per method
    pca_models = {}
    for method_key, _ in display_methods:
        feat0 = all_feats[frame_indices[0]][method_key]
        C, H, W = feat0.shape
        flat = feat0.reshape(C, -1).T.numpy()
        pca = PCA(n_components=3, random_state=42)
        pca.fit(flat)
        pca_models[method_key] = pca

    for j, (method_key, label) in enumerate(display_methods):
        col = j + 1
        axes[0, col].set_title('') # Will be set in first frame row

        for i, fi in enumerate(frame_indices):
            row = i
            feat = all_feats[fi][method_key]
            C, H, W = feat.shape

            # Use PCA fitted on first frame for consistency
            rgb = pca_colorize(feat, pca_model=pca_models[method_key])
            axes[row, col].imshow(rgb)

            m = compute_metrics(feat)
            if i == 0:
                axes[row, col].set_title(f"{label}\n({C}d@{H}×{W})\nCV={m['spatial_cv']:.3f} EffDim={m['eff_dim']:.0f}",
                                          fontsize=9)
            else:
                cos = compute_view_consistency(all_feats[frame_indices[0]][method_key], feat)
                axes[row, col].set_title(f"Frame {fi} (cos={cos:.3f})", fontsize=9)
            axes[row, col].axis('off')

    # Add a bottom row showing downsampled versions at 20×35 (coarse scale)
    bottom_row = n_frames
    coarse_size = (20, 35)
    axes[bottom_row, 0].text(0.5, 0.5, f'Downsampled\nto {coarse_size[0]}×{coarse_size[1]}\n(coarse)',
                              ha='center', va='center', fontsize=12, fontweight='bold',
                              transform=axes[bottom_row, 0].transAxes)
    axes[bottom_row, 0].axis('off')

    for j, (method_key, label) in enumerate(display_methods):
        col = j + 1
        feat = all_feats[frame_indices[0]][method_key]
        feat_ds = F.adaptive_avg_pool2d(feat.unsqueeze(0), coarse_size).squeeze(0)
        rgb = pca_colorize(feat_ds)
        axes[bottom_row, col].imshow(rgb, interpolation='nearest')
        axes[bottom_row, col].set_title(f"@ {coarse_size[0]}×{coarse_size[1]}", fontsize=9)
        axes[bottom_row, col].axis('off')

    plt.tight_layout()
    out_path = os.path.join(args.output, 'feature_comparison.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # ─── Also save a zoomed-in detail comparison ───
    print("[Generating detail comparison...]")
    fig2, axes2 = plt.subplots(2, n_methods, figsize=(4 * n_methods, 8))
    if n_methods == 1:
        axes2 = axes2[:, np.newaxis]

    fi = frame_indices[0]
    for j, (method_key, label) in enumerate(display_methods):
        feat = all_feats[fi][method_key]
        C, H, W = feat.shape
        rgb = pca_colorize(feat, pca_model=pca_models[method_key])

        # Full image
        axes2[0, j].imshow(rgb)
        m = compute_metrics(feat)
        axes2[0, j].set_title(f"{label}\nCV={m['spatial_cv']:.3f}", fontsize=9)
        axes2[0, j].axis('off')

        # Crop center 1/4
        ch, cw = H // 4, W // 4
        crop = rgb[ch:ch*3, cw:cw*3]
        axes2[1, j].imshow(crop, interpolation='nearest')
        axes2[1, j].set_title("Center crop (2× zoom)", fontsize=9)
        axes2[1, j].axis('off')

    plt.suptitle(f"Frame {fi}: Feature Detail Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path2 = os.path.join(args.output, 'feature_detail.png')
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path2}")

    # ─── L2-normalized view ───
    print("[Generating L2-normalized comparison...]")
    fig3, axes3 = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
    if n_methods == 1:
        axes3 = [axes3]

    for j, (method_key, label) in enumerate(display_methods):
        feat = all_feats[frame_indices[0]][method_key]
        # L2 normalize per pixel
        feat_n = F.normalize(feat, dim=0)
        rgb = pca_colorize(feat_n)
        axes3[j].imshow(rgb)
        m = compute_metrics(feat_n)
        axes3[j].set_title(f"{label}\n(L2 normed)\nCV={m['spatial_cv']:.3f}", fontsize=9)
        axes3[j].axis('off')

    plt.suptitle("L2-Normalized Features (per-pixel)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path3 = os.path.join(args.output, 'feature_l2normed.png')
    fig3.savefig(out_path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path3}")

    print(f"\nAll outputs saved to {args.output}/")


if __name__ == '__main__':
    main()
