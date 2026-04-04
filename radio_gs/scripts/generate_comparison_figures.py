#!/usr/bin/env python3
"""Generate side-by-side comparison figures across RADIO-GS methods.

Produces publication-quality comparison grids for:
  1. Feature PCA: GT vs Method1 vs Method2 vs ... (shared SVD)
  2. Depth estimation comparison (all methods + GT)
  3. Segmentation comparison (all methods + GT)
  4. Grounding heatmap comparison

Usage:
    python radio_gs/scripts/generate_comparison_figures.py \
        --output_dir /root/results/comparison \
        --num_views 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from radio_gs.config import load_config
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.screen_refiner import ScreenSpaceRefiner
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

device = torch.device("cuda")

# Methods to compare
DEFAULT_METHODS = [
    {
        "name": "V11 (Ours)",
        "config": "radio_gs/configs/replica_explicit_v11.yaml",
        "checkpoint": "output/radio_gs/room0_explicit_v11/checkpoints/best.pth",
    },
    {
        "name": "Baseline (No Refiner/Sharp)",
        "config": "radio_gs/configs/baseline_f3dgs.yaml",
        "checkpoint": "output/radio_gs/room0_baseline_f3dgs/checkpoints/best.pth",
    },
    {
        "name": "Ablation (FeatSharp only)",
        "config": "radio_gs/configs/ablation_no_refiner.yaml",
        "checkpoint": "output/radio_gs/room0_ablation_no_refiner/checkpoints/best.pth",
    },
]

from radio_gs.replica_constants import REPLICA_CLASSES, SEG_COLORS


# ── Pipeline loading ──────────────────────────────────────────────────────────

def load_pipeline(config_path, checkpoint_path):
    """Load the full RADIO-GS rendering pipeline."""
    config = load_config(config_path)
    model = ExplicitFeatureGaussian(latent_dim=getattr(config, "latent_dim", 64))
    ply_path = getattr(config, "ply_path", "")
    if ply_path:
        model.load_from_ply(ply_path)
    model = model.to(device).eval()

    codec = HCDCodec(
        input_dim=getattr(config, "radio_feature_dim", 1280),
        bottleneck_dim=getattr(config, "bottleneck_dim", 64),
        dual_stream=getattr(config, "dual_stream", True),
    ).to(device).eval()

    fH = getattr(config, "feature_height", 30)
    fW = getattr(config, "feature_width", 40)
    renderer = FeatureFieldRenderer(
        image_height=fH, image_width=fW,
        fx=getattr(config, "fx", 320.0) * fW / getattr(config, "image_width", 640),
        fy=getattr(config, "fy", 320.0) * fH / getattr(config, "image_height", 480),
        cx=getattr(config, "cx", 319.5) * fW / getattr(config, "image_width", 640),
        cy=getattr(config, "cy", 239.5) * fH / getattr(config, "image_height", 480),
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=getattr(config, "use_2dgs", False),
    ).to(device)

    sharpener = FeatSharp3D(
        mode=getattr(config, "featsharp_mode", "analytical"),
        feature_dim=getattr(config, "latent_dim", 64),
        strength=getattr(config, "featsharp_strength", 0.3),
    ).to(device).eval()

    refiner = None
    if getattr(config, "use_refiner", False):
        extra_ch = 0
        if getattr(config, "refiner_rgb_guide", False):
            extra_ch += 3
        if getattr(config, "refiner_depth_guide", False):
            extra_ch += 3 if getattr(config, "refiner_depth_grad", False) else 1
        refiner = ScreenSpaceRefiner(
            latent_dim=getattr(config, "latent_dim", 64),
            hidden_dim=getattr(config, "refiner_hidden_dim", 128),
            num_blocks=getattr(config, "refiner_num_blocks", 4),
            dropout=getattr(config, "refiner_dropout", 0.1),
            extra_channels=extra_ch,
            norm_type=getattr(config, "refiner_norm_type", "gn"),
        ).to(device).eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
    if "sharpener_state_dict" in ckpt:
        sharpener.load_state_dict(ckpt["sharpener_state_dict"], strict=False)
    if refiner is not None and "refiner_state_dict" in ckpt:
        refiner.load_state_dict(ckpt["refiner_state_dict"], strict=False)

    return model, codec, renderer, sharpener, refiner, config


def render_features(model, codec, renderer, sharpener, refiner, config, viewmat):
    """Render decoded 1280d features for a single view."""
    with torch.no_grad():
        if getattr(config, "self_guided", False):
            vm = viewmat if viewmat.dim() == 3 else viewmat.unsqueeze(0)
            result = renderer.render_features_and_rgb(model, vm)
            latent = result["feature_map"]
            rgb_guide = result["rgb"]
        else:
            vm = viewmat if viewmat.dim() == 3 else viewmat.unsqueeze(0)
            result = renderer.render_features_batch(model, vm)
            latent = result["feature_map"]
            rgb_guide = None
        latent = sharpener(latent)
        if refiner is not None:
            latent = refiner(latent, guide=rgb_guide)
        decoded = codec.decoder(latent)
    return decoded  # [1, 1280, H, W]


# ── Visualization helpers ─────────────────────────────────────────────────────

def shared_pca_colorize(features_list, n_components=3):
    """Apply shared PCA across multiple feature maps."""
    all_flat, shapes = [], []
    for feat in features_list:
        C, H, W = feat.shape
        shapes.append((H, W))
        all_flat.append(feat.reshape(C, -1).T.cpu().numpy())
    stacked = np.concatenate(all_flat, axis=0)
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:n_components]
    results = []
    offset = 0
    for (H, W) in shapes:
        n = H * W
        proj = centered[offset:offset + n] @ basis.T
        offset += n
        for c in range(n_components):
            vmin, vmax = proj[:, c].min(), proj[:, c].max()
            if vmax - vmin > 1e-6:
                proj[:, c] = (proj[:, c] - vmin) / (vmax - vmin)
            else:
                proj[:, c] = 0.5
        results.append((proj.reshape(H, W, n_components) * 255).astype(np.uint8))
    return results


def depth_to_colormap(depth, vmin=None, vmax=None):
    d = depth.copy().astype(np.float32)
    if vmin is None:
        vmin = d[d > 0.01].min() if (d > 0.01).any() else 0
    if vmax is None:
        vmax = d.max()
    d = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
    colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def seg_overlay(rgb_img, seg_map, alpha=0.5):
    """Create segmentation overlay on RGB image."""
    h, w = seg_map.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in SEG_COLORS.items():
        mask = seg_map == cid
        overlay[mask] = color
    if rgb_img is not None:
        rgb_rs = cv2.resize(rgb_img, (w, h))
        blended = (alpha * overlay + (1 - alpha) * rgb_rs).astype(np.uint8)
    else:
        blended = overlay
    return blended


def cosine_heatmap(cos_map):
    cos_clipped = np.clip(cos_map, 0, 1)
    hm = cv2.applyColorMap((cos_clipped * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)


# ── Downstream probes ─────────────────────────────────────────────────────────

def train_linear_probe(features, targets, lr=0.01, steps=500):
    """Train a simple linear probe and return it."""
    C = features.shape[1]
    probe = torch.nn.Linear(C, 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
    targets_flat = targets.reshape(-1, 1)
    valid = targets_flat.squeeze() > 0.01
    for _ in range(steps):
        pred = probe(features_flat[valid])
        loss = F.l1_loss(pred, targets_flat[valid])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return probe


def predict_depth(probe, features):
    """Predict depth from features using trained probe."""
    C = features.shape[1]
    H, W = features.shape[2], features.shape[3]
    with torch.no_grad():
        flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        pred = probe(flat).reshape(1, H, W)
    return pred.cpu().numpy()[0]


def train_seg_probe(features, labels, num_classes=101, lr=0.01, steps=500):
    """Train segmentation classifier."""
    C = features.shape[1]
    clf = torch.nn.Conv2d(C, num_classes, 1).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(steps):
        logits = clf(features)
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.cross_entropy(logits.squeeze(0), labels.squeeze(0).long())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return clf


def predict_seg(clf, features, target_size=None):
    """Predict segmentation from features."""
    with torch.no_grad():
        logits = clf(features)
        if target_size is not None and logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, target_size, mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1)
    return pred.cpu().numpy()[0]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_val_data(scene="room_0", num_views=5):
    """Load validation poses, GT features, depth, semantics, RGB."""
    import os

    base = f"dataset/{scene}/Sequence_2"
    feat_base = f"output/radio_features_1280d/{scene}/Sequence_2/backbone"

    # Load poses (each line = 16 floats for one 4x4 c2w matrix)
    pose_file = os.path.join(base, "traj_w_c.txt")
    poses = []
    with open(pose_file, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                c2w = np.array(vals).reshape(4, 4)
                w2c = np.linalg.inv(c2w)
                poses.append(w2c.astype(np.float32))

    # Sample evenly
    total = len(poses)
    indices = np.linspace(0, total - 1, num_views, dtype=int)

    data = []
    for idx in indices:
        item = {"idx": idx, "viewmat": torch.from_numpy(poses[idx]).to(device)}

        # GT features
        feat_path = os.path.join(feat_base, f"rgb_{idx}.pt")
        if os.path.exists(feat_path):
            item["gt_features"] = torch.load(feat_path, map_location=device).float()
            if item["gt_features"].dim() == 3:
                item["gt_features"] = item["gt_features"].unsqueeze(0)

        # Depth
        depth_path = os.path.join(base, "depth", f"depth_{idx}.png")
        if os.path.exists(depth_path):
            d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            item["gt_depth"] = d

        # Semantics
        sem_path = os.path.join(base, "semantic_class", f"semantic_class_{idx}.png")
        if os.path.exists(sem_path):
            item["gt_seg"] = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)

        # RGB
        rgb_path = os.path.join(base, "rgb", f"rgb_{idx}.png")
        if os.path.exists(rgb_path):
            item["rgb"] = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

        data.append(item)
    return data


# ── Main figure generation ────────────────────────────────────────────────────

def generate_pca_comparison(methods_data, val_data, output_dir):
    """Generate PCA comparison grid: rows=views, cols=GT + methods."""
    import matplotlib.pyplot as plt

    num_views = len(val_data)
    num_methods = len(methods_data)
    ncols = 1 + num_methods  # GT + methods

    fig, axes = plt.subplots(num_views, ncols, figsize=(3 * ncols, 3 * num_views))
    if num_views == 1:
        axes = axes[np.newaxis, :]

    for vi, vd in enumerate(val_data):
        # Collect all feature maps for shared PCA
        all_feats = []
        gt_feat = vd.get("gt_features")
        if gt_feat is not None:
            all_feats.append(gt_feat.squeeze(0))
        for md in methods_data:
            all_feats.append(md["rendered"][vi].squeeze(0))

        pca_imgs = shared_pca_colorize(all_feats)
        up = lambda img: cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)

        col = 0
        if gt_feat is not None:
            axes[vi, col].imshow(up(pca_imgs[col]))
            if vi == 0:
                axes[vi, col].set_title("GT RADIO", fontsize=10, fontweight="bold")
            col_offset = 1
        else:
            col_offset = 0

        for mi, md in enumerate(methods_data):
            c = mi + col_offset
            axes[vi, c].imshow(up(pca_imgs[c]))
            # Compute cosine sim
            cos = F.cosine_similarity(
                gt_feat.float().flatten(2), md["rendered"][vi].float().flatten(2), dim=1
            ).mean().item() if gt_feat is not None else 0
            if vi == 0:
                axes[vi, c].set_title(f"{md['name']}", fontsize=9, fontweight="bold")
            axes[vi, c].set_ylabel(f"cos={cos:.3f}", fontsize=8, rotation=0, labelpad=50)

        for ax in axes[vi]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Feature PCA Comparison (Novel Views)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "pca_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pca_comparison.png")


def generate_depth_comparison(methods_data, val_data, output_dir):
    """Generate depth comparison grid."""
    import matplotlib.pyplot as plt

    # Train depth probes on GT features first
    print("  Training depth probes...")
    probes = {}
    # Use first few frames for training
    train_feats = []
    train_depths = []
    for vd in val_data[:3]:
        if "gt_features" in vd and "gt_depth" in vd:
            feat = vd["gt_features"]
            d = torch.from_numpy(vd["gt_depth"]).unsqueeze(0).to(device)
            d_rs = F.interpolate(d.unsqueeze(0), feat.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
            train_feats.append(feat)
            train_depths.append(d_rs)

    if not train_feats:
        print("  No depth data, skipping.")
        return

    train_f = torch.cat(train_feats, 0)
    train_d = torch.cat(train_depths, 0)

    # Oracle probe (GT features)
    oracle_probe = train_linear_probe(train_f, train_d, steps=800)

    # Per-method probes (rendered features)
    method_probes = []
    for md in methods_data:
        rend_feats = torch.cat([md["rendered"][i] for i in range(min(3, len(val_data)))], 0)
        mp = train_linear_probe(rend_feats, train_d[:rend_feats.shape[0]], steps=800)
        method_probes.append(mp)

    # Visualize on remaining frames
    test_views = val_data[3:] if len(val_data) > 3 else val_data
    if not test_views:
        test_views = val_data[-2:]

    num_views = len(test_views)
    ncols = 2 + len(methods_data)  # GT + Oracle + methods
    fig, axes = plt.subplots(num_views, ncols, figsize=(3 * ncols, 3 * num_views))
    if num_views == 1:
        axes = axes[np.newaxis, :]

    for vi, vd in enumerate(test_views):
        gt_depth = vd.get("gt_depth")
        if gt_depth is None:
            continue
        vmin, vmax = gt_depth[gt_depth > 0.01].min(), gt_depth.max()

        # GT depth
        axes[vi, 0].imshow(depth_to_colormap(gt_depth, vmin, vmax))
        if vi == 0:
            axes[vi, 0].set_title("GT Depth", fontsize=10, fontweight="bold")

        # Oracle
        oracle_pred = predict_depth(oracle_probe, vd["gt_features"])
        oracle_pred_rs = cv2.resize(oracle_pred, (gt_depth.shape[1], gt_depth.shape[0]))
        axes[vi, 1].imshow(depth_to_colormap(oracle_pred_rs, vmin, vmax))
        if vi == 0:
            axes[vi, 1].set_title("Oracle (GT feat)", fontsize=10, fontweight="bold")

        # Methods
        idx_in_val = val_data.index(vd)
        for mi, md in enumerate(methods_data):
            pred = predict_depth(method_probes[mi], md["rendered"][idx_in_val])
            pred_rs = cv2.resize(pred, (gt_depth.shape[1], gt_depth.shape[0]))
            axes[vi, 2 + mi].imshow(depth_to_colormap(pred_rs, vmin, vmax))
            # AbsRel
            valid = gt_depth > 0.01
            absrel = np.abs(pred_rs[valid] - gt_depth[valid]).mean() / gt_depth[valid].mean()
            if vi == 0:
                axes[vi, 2 + mi].set_title(f"{md['name']}", fontsize=9, fontweight="bold")
            axes[vi, 2 + mi].set_xlabel(f"AbsRel={absrel:.3f}", fontsize=8)

        for ax in axes[vi]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Depth Estimation Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "depth_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved depth_comparison.png")


def generate_seg_comparison(methods_data, val_data, output_dir):
    """Generate segmentation comparison grid."""
    import matplotlib.pyplot as plt

    print("  Training seg probes...")
    train_feats, train_labels = [], []
    for vd in val_data[:3]:
        if "gt_features" in vd and "gt_seg" in vd:
            feat = vd["gt_features"]
            seg = torch.from_numpy(vd["gt_seg"].astype(np.int64)).unsqueeze(0).to(device)
            train_feats.append(feat)
            train_labels.append(seg)

    if not train_feats:
        print("  No seg data, skipping.")
        return

    train_f = torch.cat(train_feats, 0)
    train_l = torch.cat(train_labels, 0)

    oracle_clf = train_seg_probe(train_f, train_l, steps=800)

    method_clfs = []
    for md in methods_data:
        rend_feats = torch.cat([md["rendered"][i] for i in range(min(3, len(val_data)))], 0)
        mc = train_seg_probe(rend_feats, train_l[:rend_feats.shape[0]], steps=800)
        method_clfs.append(mc)

    test_views = val_data[3:] if len(val_data) > 3 else val_data[-2:]
    num_views = len(test_views)
    ncols = 2 + len(methods_data)
    fig, axes = plt.subplots(num_views, ncols, figsize=(3 * ncols, 3 * num_views))
    if num_views == 1:
        axes = axes[np.newaxis, :]

    for vi, vd in enumerate(test_views):
        gt_seg = vd.get("gt_seg")
        rgb = vd.get("rgb")
        if gt_seg is None:
            continue

        axes[vi, 0].imshow(seg_overlay(rgb, gt_seg))
        if vi == 0:
            axes[vi, 0].set_title("GT Seg", fontsize=10, fontweight="bold")

        oracle_pred = predict_seg(oracle_clf, vd["gt_features"], gt_seg.shape)
        axes[vi, 1].imshow(seg_overlay(rgb, oracle_pred))
        if vi == 0:
            axes[vi, 1].set_title("Oracle", fontsize=10, fontweight="bold")

        idx_in_val = val_data.index(vd)
        for mi, md in enumerate(methods_data):
            pred = predict_seg(method_clfs[mi], md["rendered"][idx_in_val], gt_seg.shape)
            axes[vi, 2 + mi].imshow(seg_overlay(rgb, pred))
            if vi == 0:
                axes[vi, 2 + mi].set_title(f"{md['name']}", fontsize=9, fontweight="bold")

        for ax in axes[vi]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Semantic Segmentation Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "seg_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved seg_comparison.png")


def generate_cosine_comparison(methods_data, val_data, output_dir):
    """Generate cosine similarity heatmap comparison."""
    import matplotlib.pyplot as plt

    num_views = min(5, len(val_data))
    ncols = len(methods_data)
    fig, axes = plt.subplots(num_views, ncols, figsize=(3.5 * ncols, 3 * num_views))
    if num_views == 1:
        axes = axes[np.newaxis, :]

    for vi in range(num_views):
        vd = val_data[vi]
        gt = vd.get("gt_features")
        if gt is None:
            continue

        for mi, md in enumerate(methods_data):
            rend = md["rendered"][vi]
            # Compute per-pixel cosine similarity
            cos = F.cosine_similarity(gt.float(), rend.float(), dim=1).squeeze(0).cpu().numpy()
            hm = cosine_heatmap(cos)
            hm_up = cv2.resize(hm, (320, 240), interpolation=cv2.INTER_NEAREST)
            axes[vi, mi].imshow(hm_up)
            mean_cos = cos.mean()
            if vi == 0:
                axes[vi, mi].set_title(f"{md['name']}", fontsize=9, fontweight="bold")
            axes[vi, mi].set_xlabel(f"mean={mean_cos:.3f}", fontsize=8)
            axes[vi, mi].set_xticks([])
            axes[vi, mi].set_yticks([])

    fig.suptitle("Per-Pixel Cosine Similarity (GT vs Rendered)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "cosine_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved cosine_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/root/results/comparison")
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--scene", default="room_0")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only methods with existing checkpoints
    methods = []
    for m in DEFAULT_METHODS:
        if os.path.exists(m["checkpoint"]):
            methods.append(m)
            print(f"✅ Found: {m['name']} ({m['checkpoint']})")
        else:
            print(f"⏭️  Skipping: {m['name']} (checkpoint not found)")

    if not methods:
        print("No methods with checkpoints found!")
        return

    # Load validation data
    print(f"\nLoading validation data ({args.scene}, {args.num_views} views)...")
    val_data = load_val_data(args.scene, args.num_views)
    print(f"  Loaded {len(val_data)} views")

    # Render features for each method
    methods_data = []
    for m in methods:
        print(f"\nLoading {m['name']}...")
        pipeline = load_pipeline(m["config"], m["checkpoint"])
        model, codec, renderer, sharpener, refiner, config = pipeline

        rendered = []
        for vd in tqdm(val_data, desc=f"  Rendering {m['name']}"):
            feat = render_features(model, codec, renderer, sharpener, refiner, config, vd["viewmat"])
            rendered.append(feat)

        methods_data.append({"name": m["name"], "rendered": rendered})

        # Free GPU memory
        del model, codec, renderer, sharpener, refiner
        torch.cuda.empty_cache()

    # Generate comparison figures
    print("\nGenerating comparison figures...")
    generate_pca_comparison(methods_data, val_data, output_dir)
    generate_cosine_comparison(methods_data, val_data, output_dir)
    generate_depth_comparison(methods_data, val_data, output_dir)
    generate_seg_comparison(methods_data, val_data, output_dir)

    print(f"\n✅ All comparison figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
