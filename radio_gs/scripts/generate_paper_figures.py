"""Generate publication-quality figures for the RADIO-GS paper.

Produces four key figures:
  1. Grounding Summary Grid — 3 frames × (RGB + 4 query heatmaps), GT & Rendered
  2. Multi-Task Overview — 3 frames × 6 columns (RGB, PCA GT/Rend, Depth, Seg, Grounding)
  3. Feature Quality Comparison — GT PCA vs Rendered PCA vs Cosine Similarity
  4. Ablation Comparison — V9 (GT RGB guide) vs V10c (no guide) vs V11 (self-RGB)

Usage:
    python radio_gs/scripts/generate_paper_figures.py \
        --config radio_gs/configs/replica_explicit_v11.yaml \
        --output_dir /root/results/paper/ \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from radio_gs.config import load_config
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.screen_refiner import ScreenSpaceRefiner
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

# ── Constants ─────────────────────────────────────────────────────────────────

from radio_gs.replica_constants import REPLICA_CLASSES, SEG_COLORS, GROUNDING_QUERIES

ALL_GROUNDING_QUERIES = list(GROUNDING_QUERIES.keys())
SELECTED_QUERIES = ["chair", "plant", "wall", "sofa"]
FRAME_INDICES = [0, 180, 360]


# ── Pipeline loading (mirrors generate_visualizations.py) ─────────────────────

def load_pipeline(config_path, checkpoint_path, device):
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
    iH = getattr(config, "image_height", 480)
    iW = getattr(config, "image_width", 640)
    renderer = FeatureFieldRenderer(
        image_height=fH, image_width=fW,
        fx=getattr(config, "fx", 320.0) * fW / iW,
        fy=getattr(config, "fy", 320.0) * fH / iH,
        cx=getattr(config, "cx", 319.5) * fW / iW,
        cy=getattr(config, "cy", 239.5) * fH / iH,
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=getattr(config, "use_2dgs", False),
    ).to(device)

    sharpener = FeatSharp3D(
        mode=getattr(config, "featsharp_mode", "analytical"),
        feature_dim=getattr(config, "latent_dim", 64),
        strength=getattr(config, "featsharp_strength", 0.3),
    ).to(device).eval()

    refiner = None
    rgb_guide_enabled = getattr(config, "refiner_rgb_guide", False)
    depth_guide_enabled = getattr(config, "refiner_depth_guide", False)
    depth_grad_enabled = getattr(config, "refiner_depth_grad", False)
    if getattr(config, "use_refiner", False):
        extra_ch = 0
        if rgb_guide_enabled:
            extra_ch += 3
        if depth_guide_enabled:
            extra_ch += 3 if depth_grad_enabled else 1
        norm_type = getattr(config, "refiner_norm_type", "gn")
        refiner = ScreenSpaceRefiner(
            latent_dim=getattr(config, "latent_dim", 64),
            hidden_dim=getattr(config, "refiner_hidden_dim", 128),
            num_blocks=getattr(config, "refiner_num_blocks", 4),
            dropout=getattr(config, "refiner_dropout", 0.1),
            extra_channels=extra_ch,
            norm_type=norm_type,
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
    """Render and decode 1280d features for a single view."""
    self_guided = getattr(config, "self_guided", False)
    with torch.no_grad():
        if self_guided:
            vm = viewmat if viewmat.dim() == 3 else viewmat.unsqueeze(0)
            result = renderer.render_features_and_rgb(model, vm)
            latent = result["feature_map"]
            rgb_guide = result["rgb"]
        else:
            result = renderer.render_features_batch(model, viewmat)
            latent = result["feature_map"]
            rgb_guide = None
        latent = sharpener(latent)
        if refiner is not None:
            latent = refiner(latent, guide=rgb_guide)
        decoded = codec.decoder(latent)
    return decoded  # [1, 1280, H, W]


# ── Visualization helpers ─────────────────────────────────────────────────────

def shared_pca_colorize(features_list, n_components=3):
    """Apply shared PCA to multiple feature maps for comparable visualization."""
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
    """Convert depth map to INFERNO colormap visualization."""
    d = depth.copy().astype(np.float32)
    if vmin is None:
        vmin = d[d > 0.01].min() if (d > 0.01).any() else 0
    if vmax is None:
        vmax = d.max()
    d = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
    colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def seg_to_color(seg_map, color_map=None):
    """Convert class ID map to RGB visualization."""
    if color_map is None:
        color_map = SEG_COLORS
    H, W = seg_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, color in color_map.items():
        mask = seg_map == cid
        if mask.any():
            rgb[mask] = color
    return rgb


def upscale(img, scale, interp=cv2.INTER_NEAREST):
    """Upscale image by integer factor."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=interp)


# ── SigLIP2 grounding ────────────────────────────────────────────────────────

def load_siglip2_projection(projection_weights, device):
    """Load SigLIP2 feature projection model."""
    from timm.models.vision_transformer import Block

    class SigLIP2FeatureProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(*[
                Block(1280, num_heads=16, init_values=1e-5) for _ in range(2)
            ])
            self.mlp_fc1 = nn.Linear(1280, 1520)
            self.mlp_final = nn.Sequential(
                nn.LayerNorm(1520), nn.GELU(), nn.Linear(1520, 1536),
            )

        def forward(self, x):
            x = self.blocks(x)
            x = self.mlp_fc1(x)
            return self.mlp_final(x)

    proj = SigLIP2FeatureProjection()
    proj.load_state_dict(torch.load(projection_weights, map_location="cpu"))
    return proj.to(device).half().eval()


def compute_grounding_heatmaps(features_1280, proj_model, text_emb):
    """Compute text grounding heatmaps.

    Args:
        features_1280: [1, 1280, H, W]
        proj_model: SigLIP2 projection
        text_emb: [K, 1536] normalized text embeddings

    Returns:
        [K, H, W] similarity heatmaps (normalized to 0-1 per query)
    """
    B, C, H, W = features_1280.shape
    feat_flat = features_1280.reshape(B, C, H * W).permute(0, 2, 1)
    with torch.no_grad():
        siglip = proj_model(feat_flat.half())
    siglip = F.normalize(siglip, dim=-1).squeeze(0)  # [HW, 1536]
    sim = text_emb @ siglip.T  # [K, HW]
    return sim.float().reshape(-1, H, W)


def heatmap_overlay(rgb, heatmap_raw, alpha=0.55):
    """Create a JET heatmap overlay on RGB. heatmap_raw is [H,W] floats."""
    h, w = heatmap_raw.shape
    vmin, vmax = heatmap_raw.min(), heatmap_raw.max()
    if vmax - vmin > 1e-6:
        norm = (heatmap_raw - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(heatmap_raw)
    jet = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (w, h))
    blended = (alpha * jet.astype(np.float32) + (1 - alpha) * rgb_resized.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── Linear probe helpers ─────────────────────────────────────────────────────

def train_depth_probe(features, depth_dir, indices, fH, fW, device):
    """Train a linear depth probe."""
    train_X, train_Y = [], []
    for feat, i in zip(features, indices):
        dpath = Path(depth_dir) / f"depth_{i}.png"
        if not dpath.exists():
            continue
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        d = torch.from_numpy(d.astype(np.float32) / 1000.0)
        d = F.interpolate(d[None, None], (fH, fW), mode="bilinear",
                          align_corners=False).squeeze()
        C = feat.shape[0]
        f = feat
        if f.shape[1:] != (fH, fW):
            f = F.interpolate(f[None], (fH, fW), mode="bilinear",
                              align_corners=False).squeeze(0)
        valid = d > 0.01
        if valid.sum() < 10:
            continue
        train_X.append(f.reshape(C, -1).T[valid.reshape(-1)])
        train_Y.append(d.reshape(-1)[valid.reshape(-1)])

    train_X = torch.cat(train_X, 0).to(device)
    train_Y = torch.cat(train_Y, 0).to(device)
    probe = nn.Linear(train_X.shape[1], 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for _ in range(100):
        pred = probe(train_X).squeeze()
        loss = F.l1_loss(pred, train_Y)
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    return probe


def train_seg_probe(features, sem_dir, indices, fH, fW, device):
    """Train a linear segmentation probe."""
    train_X, train_Y = [], []
    for feat, i in zip(features, indices):
        spath = Path(sem_dir) / f"semantic_class_{i}.png"
        if not spath.exists():
            continue
        sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        sem = torch.from_numpy(sem.astype(np.int64))
        sem = F.interpolate(sem.float()[None, None], (fH, fW),
                            mode="nearest").squeeze().long()
        C = feat.shape[0]
        f = feat
        if f.shape[1:] != (fH, fW):
            f = F.interpolate(f[None], (fH, fW), mode="bilinear",
                              align_corners=False).squeeze(0)
        train_X.append(f.reshape(C, -1).T)
        train_Y.append(sem.reshape(-1))

    train_X = torch.cat(train_X, 0).to(device)
    train_Y = torch.cat(train_Y, 0).to(device)
    n_classes = int(train_Y.max().item()) + 1
    probe = nn.Linear(train_X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for _ in range(200):
        logits = probe(train_X)
        loss = F.cross_entropy(logits, train_Y)
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    return probe, n_classes


def predict_depth(probe, feat, fH, fW, device):
    """Predict depth from features using linear probe."""
    C = feat.shape[0]
    f = feat.to(device)
    if f.shape[1:] != (fH, fW):
        f = F.interpolate(f[None], (fH, fW), mode="bilinear",
                          align_corners=False).squeeze(0)
    with torch.no_grad():
        pred = probe(f.reshape(C, -1).T).squeeze().reshape(fH, fW)
    return pred.cpu().numpy()


def predict_seg(probe, feat, fH, fW, device):
    """Predict segmentation from features using linear probe."""
    C = feat.shape[0]
    f = feat.to(device)
    if f.shape[1:] != (fH, fW):
        f = F.interpolate(f[None], (fH, fW), mode="bilinear",
                          align_corners=False).squeeze(0)
    with torch.no_grad():
        pred = probe(f.reshape(C, -1).T).argmax(1).reshape(fH, fW)
    return pred.cpu().numpy()


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_poses(pose_file):
    """Load camera poses from traj_w_c.txt (c2w), return w2c matrices."""
    all_c2w = np.loadtxt(str(pose_file)).reshape(-1, 4, 4).astype(np.float32)
    return np.linalg.inv(all_c2w)


def load_gt_features(feat_dir, idx, fH, fW):
    """Load ground truth 1280d features for a frame index."""
    fpath = Path(feat_dir) / f"rgb_{idx}.pt"
    feat = torch.load(str(fpath), map_location="cpu").float()
    if feat.dim() == 2:
        feat = feat.reshape(fH, fW, -1).permute(2, 0, 1)
    return feat


def load_rgb(rgb_dir, idx):
    """Load RGB image."""
    rgb = cv2.imread(str(Path(rgb_dir) / f"rgb_{idx}.png"))
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


# ── Figure generators ─────────────────────────────────────────────────────────

def generate_grounding_grid(
    gt_feats, rend_feats, rgb_images, frame_indices,
    proj_model, text_data, device, output_path,
    fH=30, fW=40, scale=8,
):
    """Figure 1: Grounding Summary Grid.

    Layout: 6 rows × 5 columns
      For each of 3 frames: GT row + Rendered row
      Columns: RGB | query1 heatmap | query2 | query3 | query4
    """
    queries = SELECTED_QUERIES
    all_queries = text_data["queries"]
    all_emb = text_data["embeddings"].to(device).half()
    q2i = {q: i for i, q in enumerate(all_queries)}
    active_emb = torch.stack([all_emb[q2i[q]] for q in queries if q in q2i])
    active_names = [q for q in queries if q in q2i]

    n_frames = len(frame_indices)
    n_cols = 1 + len(active_names)  # RGB + queries
    n_rows = n_frames * 2  # GT + Rendered per frame

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.0),
        gridspec_kw={"wspace": 0.03, "hspace": 0.08},
    )

    for fi in range(n_frames):
        rgb = rgb_images[fi]
        rgb_small = cv2.resize(rgb, (fW, fH))
        rgb_up = upscale(rgb_small, scale, cv2.INTER_LINEAR)

        gt_f = gt_feats[fi].unsqueeze(0).to(device)
        rend_f = rend_feats[fi].unsqueeze(0).to(device)

        gt_hm = compute_grounding_heatmaps(gt_f, proj_model, active_emb)
        rend_hm = compute_grounding_heatmaps(rend_f, proj_model, active_emb)

        gt_row = fi * 2
        rend_row = fi * 2 + 1

        # RGB column
        axes[gt_row, 0].imshow(rgb_up)
        axes[gt_row, 0].set_ylabel(f"Frame {frame_indices[fi]}\nGT",
                                    fontsize=9, fontweight="bold")
        axes[rend_row, 0].imshow(rgb_up)
        axes[rend_row, 0].set_ylabel("Rendered", fontsize=9, fontweight="bold")

        for qi, qname in enumerate(active_names):
            col = qi + 1

            gt_overlay = heatmap_overlay(rgb, gt_hm[qi].cpu().numpy())
            gt_overlay_up = upscale(gt_overlay, scale, cv2.INTER_LINEAR)
            axes[gt_row, col].imshow(gt_overlay_up)

            rend_overlay = heatmap_overlay(rgb, rend_hm[qi].cpu().numpy())
            rend_overlay_up = upscale(rend_overlay, scale, cv2.INTER_LINEAR)
            axes[rend_row, col].imshow(rend_overlay_up)

            if fi == 0:
                axes[gt_row, col].set_title(f'"{qname}"', fontsize=10,
                                            fontweight="bold")

    # Column header for RGB
    axes[0, 0].set_title("RGB", fontsize=10, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Text Grounding Heatmaps: GT vs Rendered Features",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {output_path}")


def generate_multitask_overview(
    gt_feats, rend_feats, rgb_images, frame_indices,
    proj_model, text_data,
    depth_dir, sem_dir,
    device, output_path,
    fH=30, fW=40, scale=8,
):
    """Figure 2: Multi-Task Overview.

    3 rows × 6 columns: RGB | PCA GT | PCA Rendered | Depth | Segmentation | Grounding(chair)
    """
    n_frames = len(frame_indices)

    # PCA across all GT+Rendered features (shared basis)
    all_for_pca = [gt_feats[i] for i in range(n_frames)] + \
                  [rend_feats[i] for i in range(n_frames)]
    pca_imgs = shared_pca_colorize(all_for_pca, n_components=3)
    gt_pcas = pca_imgs[:n_frames]
    rend_pcas = pca_imgs[n_frames:]

    # Depth probe: train on a few rendered features
    train_depth_indices = list(range(0, 100, 5))  # 20 frames
    train_depth_feats = []
    gt_feat_dir = Path(f"output/radio_features_1280d/room_0/Sequence_2/backbone")
    for i in train_depth_indices:
        try:
            f = load_gt_features(str(gt_feat_dir), i, fH, fW)
            train_depth_feats.append(f)
        except Exception:
            train_depth_indices = [idx for idx in train_depth_indices
                                   if idx != i]
    if train_depth_feats:
        depth_probe = train_depth_probe(
            train_depth_feats, depth_dir, train_depth_indices, fH, fW, device)
    else:
        depth_probe = None

    # Seg probe
    train_seg_feats = train_depth_feats
    if train_seg_feats:
        seg_probe, _ = train_seg_probe(
            train_seg_feats, sem_dir, train_depth_indices, fH, fW, device)
    else:
        seg_probe = None

    # Grounding for "chair"
    all_queries = text_data["queries"]
    all_emb = text_data["embeddings"].to(device).half()
    q2i = {q: i for i, q in enumerate(all_queries)}
    chair_emb = all_emb[q2i["chair"]].unsqueeze(0)

    fig, axes = plt.subplots(
        n_frames, 6,
        figsize=(6 * 2.6, n_frames * 2.0),
        gridspec_kw={"wspace": 0.03, "hspace": 0.08},
    )
    if n_frames == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["RGB", "PCA (GT)", "PCA (Rendered)", "Depth",
                   "Segmentation", 'Grounding ("chair")']

    for fi in range(n_frames):
        idx = frame_indices[fi]
        rgb = rgb_images[fi]
        rgb_small = cv2.resize(rgb, (fW, fH))
        rgb_up = upscale(rgb_small, scale, cv2.INTER_LINEAR)

        # Col 0: RGB
        axes[fi, 0].imshow(rgb_up)

        # Col 1: PCA GT
        axes[fi, 1].imshow(upscale(gt_pcas[fi], scale))

        # Col 2: PCA Rendered
        axes[fi, 2].imshow(upscale(rend_pcas[fi], scale))

        # Col 3: Depth
        dpath = Path(depth_dir) / f"depth_{idx}.png"
        if dpath.exists() and depth_probe is not None:
            d_raw = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
            gt_d = d_raw.astype(np.float32) / 1000.0
            gt_d_f = cv2.resize(gt_d, (fW, fH), interpolation=cv2.INTER_LINEAR)
            pred_d = predict_depth(depth_probe, rend_feats[fi], fH, fW, device)
            # Show GT depth
            depth_vis = depth_to_colormap(gt_d_f)
            axes[fi, 3].imshow(upscale(depth_vis, scale))
        else:
            axes[fi, 3].imshow(np.zeros((fH * scale, fW * scale, 3), dtype=np.uint8))

        # Col 4: Segmentation
        spath = Path(sem_dir) / f"semantic_class_{idx}.png"
        if spath.exists() and seg_probe is not None:
            sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
            gt_sem = cv2.resize(sem, (fW, fH), interpolation=cv2.INTER_NEAREST)
            seg_vis = seg_to_color(gt_sem)
            seg_blend = (0.6 * seg_vis.astype(np.float32) +
                         0.4 * rgb_small.astype(np.float32))
            axes[fi, 4].imshow(upscale(np.clip(seg_blend, 0, 255).astype(np.uint8), scale))
        else:
            axes[fi, 4].imshow(np.zeros((fH * scale, fW * scale, 3), dtype=np.uint8))

        # Col 5: Grounding (chair)
        rend_f = rend_feats[fi].unsqueeze(0).to(device)
        hm = compute_grounding_heatmaps(rend_f, proj_model, chair_emb)
        overlay = heatmap_overlay(rgb, hm[0].cpu().numpy())
        axes[fi, 5].imshow(upscale(overlay, scale, cv2.INTER_LINEAR))

        axes[fi, 0].set_ylabel(f"Frame {idx}", fontsize=9, fontweight="bold")

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("RADIO-GS: Multi-Task Novel View Synthesis",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {output_path}")


def generate_feature_quality(
    gt_feat, rend_feat, rgb, frame_idx, output_path,
    fH=30, fW=40, scale=8,
):
    """Figure 3: Feature Quality Comparison.

    Side-by-side: GT PCA | Rendered PCA | Cosine Similarity Map
    With quantitative metric overlay.
    """
    pca_imgs = shared_pca_colorize([gt_feat, rend_feat], n_components=3)
    gt_pca, rend_pca = pca_imgs

    cos = F.cosine_similarity(
        rend_feat.reshape(-1, fH * fW),
        gt_feat.reshape(-1, fH * fW),
        dim=0,
    ).reshape(fH, fW).numpy()
    mean_cos = cos.mean()

    # Cosine similarity heatmap
    cos_clipped = np.clip(cos, 0, 1)
    cos_colored = cv2.applyColorMap((cos_clipped * 255).astype(np.uint8),
                                     cv2.COLORMAP_JET)
    cos_colored = cv2.cvtColor(cos_colored, cv2.COLOR_BGR2RGB)

    rgb_small = cv2.resize(rgb, (fW, fH))

    fig, axes = plt.subplots(
        1, 4,
        figsize=(4 * 3.2, 2.8),
        gridspec_kw={"wspace": 0.04},
    )

    axes[0].imshow(upscale(rgb_small, scale, cv2.INTER_LINEAR))
    axes[0].set_title("Input RGB", fontsize=10, fontweight="bold")

    axes[1].imshow(upscale(gt_pca, scale))
    axes[1].set_title("GT Feature (PCA)", fontsize=10, fontweight="bold")

    axes[2].imshow(upscale(rend_pca, scale))
    axes[2].set_title("Rendered Feature (PCA)", fontsize=10, fontweight="bold")

    im = axes[3].imshow(upscale(cos_colored, scale))
    axes[3].set_title(f"Cosine Similarity\n(mean = {mean_cos:.4f})",
                      fontsize=10, fontweight="bold")

    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="Cosine Sim")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"Feature Quality — Frame {frame_idx}  |  "
                 f"Mean Cosine Similarity: {mean_cos:.4f}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {output_path}")


def generate_ablation_figure(
    configs_and_names, frame_indices, rgb_images,
    val_poses, gt_feats_list, device, output_path,
    fH=30, fW=40, scale=8,
):
    """Figure 4: Ablation Comparison.

    Rows: one per frame
    Columns: RGB | V9 PCA | V10c PCA | V11 PCA | V9 cos | V10c cos | V11 cos
    If a model is unavailable, show placeholder.
    """
    n_frames = len(frame_indices)

    # Render features from each model variant
    variant_feats = {}  # name -> list of [C, H, W] tensors per frame
    variant_available = {}

    for cfg_path, ckpt_path, name in configs_and_names:
        if Path(cfg_path).exists() and Path(ckpt_path).exists():
            try:
                print(f"    Loading {name} from {ckpt_path}...")
                m, c, r, s, ref, cfg = load_pipeline(cfg_path, ckpt_path, device)
                feats = []
                for fi, fidx in enumerate(frame_indices):
                    pose = torch.from_numpy(val_poses[fidx:fidx + 1]).to(device)
                    dec = render_features(m, c, r, s, ref, cfg, pose)
                    feats.append(dec.squeeze(0).cpu())
                variant_feats[name] = feats
                variant_available[name] = True
                # Free GPU memory
                del m, c, r, s, ref
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    ⚠ Failed to load {name}: {e}")
                variant_available[name] = False
        else:
            print(f"    ⚠ {name} checkpoint not found, using placeholder")
            variant_available[name] = False

    available_names = [n for _, _, n in configs_and_names if variant_available.get(n, False)]

    if len(available_names) == 0:
        print("    No ablation models available — creating placeholder figure")
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.text(0.5, 0.5, "Ablation comparison: models not available",
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"  Saved placeholder {output_path}")
        return

    # Build shared PCA across all variants + GT
    all_pca_input = []
    pca_labels = []  # (variant_name, frame_idx_within_variant)
    for fi in range(n_frames):
        all_pca_input.append(gt_feats_list[fi])
        pca_labels.append(("GT", fi))
    for name in available_names:
        for fi in range(n_frames):
            all_pca_input.append(variant_feats[name][fi])
            pca_labels.append((name, fi))

    pca_imgs = shared_pca_colorize(all_pca_input, n_components=3)

    # Organize results
    pca_by_variant = {"GT": []}
    cos_by_variant = {}
    idx = 0
    for fi in range(n_frames):
        pca_by_variant["GT"].append(pca_imgs[idx])
        idx += 1
    for name in available_names:
        pca_by_variant[name] = []
        cos_by_variant[name] = []
        for fi in range(n_frames):
            pca_by_variant[name].append(pca_imgs[idx])
            # Compute cosine similarity
            cos = F.cosine_similarity(
                variant_feats[name][fi].reshape(-1, fH * fW),
                gt_feats_list[fi].reshape(-1, fH * fW),
                dim=0,
            ).reshape(fH, fW).numpy()
            cos_by_variant[name].append(cos)
            idx += 1

    # Layout: n_frames rows × (1 + len(available) * 2) columns
    # Cols: RGB | (PCA_v | CosSim_v) for each variant
    n_cols = 1 + len(available_names) * 2
    fig, axes = plt.subplots(
        n_frames, n_cols,
        figsize=(n_cols * 2.4, n_frames * 2.0),
        gridspec_kw={"wspace": 0.03, "hspace": 0.08},
    )
    if n_frames == 1:
        axes = axes[np.newaxis, :]

    for fi in range(n_frames):
        idx_frame = frame_indices[fi]
        rgb = rgb_images[fi]
        rgb_small = cv2.resize(rgb, (fW, fH))
        axes[fi, 0].imshow(upscale(rgb_small, scale, cv2.INTER_LINEAR))
        axes[fi, 0].set_ylabel(f"Frame {idx_frame}", fontsize=9, fontweight="bold")

        col = 1
        for vi, name in enumerate(available_names):
            axes[fi, col].imshow(upscale(pca_by_variant[name][fi], scale))
            cos = cos_by_variant[name][fi]
            cos_clipped = np.clip(cos, 0, 1)
            cos_vis = cv2.applyColorMap((cos_clipped * 255).astype(np.uint8),
                                         cv2.COLORMAP_JET)
            cos_vis = cv2.cvtColor(cos_vis, cv2.COLOR_BGR2RGB)
            axes[fi, col + 1].imshow(upscale(cos_vis, scale))
            if fi == 0:
                mean_c = cos.mean()
                axes[fi, col + 1].set_title(
                    f"cos={mean_c:.3f}", fontsize=8)
            col += 2

    # Column headers
    axes[0, 0].set_title("RGB", fontsize=9, fontweight="bold")
    col = 1
    for name in available_names:
        axes[0, col].set_title(f"{name}\nPCA", fontsize=9, fontweight="bold")
        if not axes[0, col + 1].get_title():
            cos_val = cos_by_variant[name][0].mean()
            axes[0, col + 1].set_title(f"{name}\ncos={cos_val:.3f}",
                                        fontsize=9, fontweight="bold")
        else:
            # Already set above; add variant name
            existing = axes[0, col + 1].get_title()
            axes[0, col + 1].set_title(f"{name}\n{existing}",
                                        fontsize=9, fontweight="bold")
        col += 2

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Ablation: Feature Quality Across Architecture Variants",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for RADIO-GS paper")
    parser.add_argument("--config", required=True,
                        help="V11 config YAML path")
    parser.add_argument("--output_dir", default="/root/results/paper/",
                        help="Output directory for figures")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=int, default=8,
                        help="Upscale factor for feature-res images")
    parser.add_argument("--text_embeddings",
                        default="output/radio_gs/siglip2_text_embeddings_v2.pt")
    parser.add_argument("--projection_weights",
                        default="output/radio_gs/siglip2_feat_projection.pth")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = args.scale

    # ── Resolve config and checkpoint ─────────────────────────────────────
    config_path = args.config
    config = load_config(config_path)
    output_model_dir = getattr(config, "output_dir", "output/radio_gs/room0_explicit_v11")
    checkpoint_path = str(Path(output_model_dir) / "checkpoints" / "best.pth")

    scene = getattr(config, "scene", "room_0")
    scene_root = Path("dataset") / scene
    val_split = getattr(config, "val_split", "Sequence_2")
    fH = getattr(config, "feature_height", 30)
    fW = getattr(config, "feature_width", 40)

    print("=" * 60)
    print("RADIO-GS Paper Figure Generation")
    print("=" * 60)
    print(f"Config:     {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {out_dir}")
    print(f"Device:     {device}")
    print(f"Feature res: {fH}×{fW}, upscale {scale}×")
    print()

    # ── Load V11 pipeline ─────────────────────────────────────────────────
    print("[1/5] Loading V11 pipeline...")
    model, codec, renderer, sharpener, refiner, cfg = load_pipeline(
        config_path, checkpoint_path, device)

    # ── Load poses and data paths ─────────────────────────────────────────
    val_pose_file = scene_root / val_split / "traj_w_c.txt"
    val_w2c = load_poses(val_pose_file)
    n_total = len(val_w2c)

    gt_feat_dir = Path(f"output/radio_features_1280d/{scene}/{val_split}/backbone")
    rgb_dir = scene_root / val_split / "rgb"
    depth_dir = scene_root / val_split / "depth"
    sem_dir = scene_root / val_split / "semantic_class"

    # Clamp frame indices to available range
    frame_indices = [min(i, n_total - 1) for i in FRAME_INDICES]
    print(f"Using frames: {frame_indices} (of {n_total} total)")

    # ── Render V11 features for selected frames ──────────────────────────
    print("\n[2/5] Rendering V11 features for selected frames...")
    gt_feats, rend_feats, rgb_images = [], [], []
    with torch.no_grad():
        for idx in tqdm(frame_indices, desc="Rendering"):
            gt = load_gt_features(str(gt_feat_dir), idx, fH, fW)
            gt_feats.append(gt)

            pose = torch.from_numpy(val_w2c[idx:idx + 1]).to(device)
            decoded = render_features(model, codec, renderer, sharpener,
                                      refiner, cfg, pose)
            rend_feats.append(decoded.squeeze(0).cpu())

            rgb = load_rgb(str(rgb_dir), idx)
            rgb_images.append(rgb)

    # ── Load grounding resources ──────────────────────────────────────────
    print("\n[3/5] Loading SigLIP2 grounding resources...")
    text_emb_path = Path(args.text_embeddings)
    proj_path = Path(args.projection_weights)
    grounding_available = text_emb_path.exists() and proj_path.exists()
    if grounding_available:
        text_data = torch.load(str(text_emb_path), map_location="cpu")
        proj_model = load_siglip2_projection(str(proj_path), device)
        print(f"  Loaded {len(text_data['queries'])} text queries")
    else:
        print("  ⚠ Grounding resources not found — grounding panels will be skipped")
        text_data = None
        proj_model = None

    # ── Figure 1: Grounding Summary Grid ──────────────────────────────────
    print("\n[4/5] Generating figures...")
    if grounding_available:
        print("  → Figure 1: Grounding Summary Grid")
        generate_grounding_grid(
            gt_feats, rend_feats, rgb_images, frame_indices,
            proj_model, text_data, device,
            out_dir / "grounding_grid.png",
            fH=fH, fW=fW, scale=scale,
        )
    else:
        # Create placeholder
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.text(0.5, 0.5, "Grounding grid: SigLIP2 resources not available",
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(str(out_dir / "grounding_grid.png"), dpi=200,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  → Figure 1: Saved placeholder grounding_grid.png")

    # ── Figure 2: Multi-Task Overview ─────────────────────────────────────
    if grounding_available:
        print("  → Figure 2: Multi-Task Overview")
        generate_multitask_overview(
            gt_feats, rend_feats, rgb_images, frame_indices,
            proj_model, text_data,
            str(depth_dir), str(sem_dir),
            device, out_dir / "multitask_overview.png",
            fH=fH, fW=fW, scale=scale,
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.text(0.5, 0.5, "Multi-task overview: SigLIP2 resources not available",
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(str(out_dir / "multitask_overview.png"), dpi=200,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  → Figure 2: Saved placeholder multitask_overview.png")

    # ── Figure 3: Feature Quality Comparison ──────────────────────────────
    print("  → Figure 3: Feature Quality Comparison")
    generate_feature_quality(
        gt_feats[0], rend_feats[0], rgb_images[0], frame_indices[0],
        out_dir / "feature_quality.png",
        fH=fH, fW=fW, scale=scale,
    )

    # ── Figure 4: Ablation Comparison ─────────────────────────────────────
    print("  → Figure 4: Ablation Comparison")
    # Free V11 GPU memory before loading other models
    del model, codec, renderer, sharpener, refiner
    torch.cuda.empty_cache()

    ablation_configs = [
        ("radio_gs/configs/replica_explicit_v9.yaml",
         "output/radio_gs/room0_explicit_v9/checkpoints/best.pth",
         "V9 (GT-RGB)"),
        ("radio_gs/configs/replica_explicit_v10c.yaml",
         "output/radio_gs/room0_explicit_v10c/checkpoints/best.pth",
         "V10c (no guide)"),
        (config_path, checkpoint_path, "V11 (self-RGB)"),
    ]
    generate_ablation_figure(
        ablation_configs, frame_indices, rgb_images,
        val_w2c, gt_feats, device,
        out_dir / "ablation_qualitative.png",
        fH=fH, fW=fW, scale=scale,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[5/5] Verifying outputs...")
    print("=" * 60)
    expected_files = [
        "grounding_grid.png",
        "multitask_overview.png",
        "feature_quality.png",
        "ablation_qualitative.png",
    ]
    all_ok = True
    for fname in expected_files:
        fpath = out_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✓ {fname} ({size_kb:.0f} KB)")
        else:
            print(f"  ✗ {fname} — MISSING or empty")
            all_ok = False

    if all_ok:
        print(f"\nAll {len(expected_files)} figures generated successfully!")
    else:
        print("\n⚠ Some figures were not generated correctly.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
