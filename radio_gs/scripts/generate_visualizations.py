"""Generate comprehensive visualizations for RADIO-GS qualitative evaluation.

Produces publication-quality figures for:
  1. Feature PCA comparison (GT vs Rendered) + cosine similarity maps
  2. Depth estimation (GT depth, Oracle pred, Rendered pred, error maps)
  3. Semantic segmentation (GT, Oracle pred, Rendered pred overlays)
  4. Text grounding heatmaps (GT vs Rendered for selected object queries)
  5. Multi-task composite figure (all tasks in one grid per frame)

Usage:
    python radio_gs/scripts/generate_visualizations.py \
        --config radio_gs/configs/replica_explicit_v11.yaml \
        --checkpoint output/radio_gs/room0_explicit_v11/checkpoints/best.pth \
        --output_dir /root/results \
        --num_views 10
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from radio_gs.config import load_config
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.screen_refiner import ScreenSpaceRefiner
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

device = torch.device("cuda")

# Replica semantic classes
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
    """Render and decode 1280d features + geometry depth for a single view."""
    self_guided = getattr(config, "self_guided", False)
    with torch.no_grad():
        if self_guided:
            vm = viewmat if viewmat.dim() == 3 else viewmat.unsqueeze(0)
            result = renderer.render_features_and_rgb(model, vm)
            latent = result["feature_map"]
            rgb_guide = result["rgb"]
        else:
            vm = viewmat if viewmat.dim() == 3 else viewmat.unsqueeze(0)
            result = renderer.render_features_batch(model, vm)
            latent = result["feature_map"]
            rgb_guide = None

        alpha_map = result.get("alpha_map", None)

        # Render geometry depth by splatting camera-space z as 1-ch color
        geom_depth = _render_geometry_depth(model, renderer, vm)

        latent = sharpener(latent)
        if refiner is not None:
            latent = refiner(latent, guide=rgb_guide)
        decoded = codec.decoder(latent)
    return decoded, geom_depth, alpha_map


def _render_geometry_depth(model, renderer, viewmat):
    """Render per-pixel geometry depth by splatting camera-space z-coordinates."""
    from gsplat import rasterization_2dgs, rasterization

    means = model.get_xyz()
    quats = model.get_rotation()
    scales = model.get_scaling()
    opacs = model.get_opacity().squeeze(-1)

    # Camera-space z-depth per Gaussian
    vm = viewmat[0] if viewmat.dim() == 3 else viewmat
    R, t = vm[:3, :3], vm[:3, 3]
    z_depth = (means @ R.T + t)[:, 2:3]  # [N, 1]

    if renderer.use_2dgs and scales.shape[-1] == 2:
        pad = torch.full((scales.shape[0], 1), -10.0,
                         device=scales.device, dtype=scales.dtype)
        scales = torch.cat([scales, pad], dim=-1)

    Ks = renderer.K.unsqueeze(0)
    bg = torch.zeros(1, 1, device=means.device)
    raster_fn = rasterization_2dgs if renderer.use_2dgs else rasterization

    if renderer.use_2dgs:
        renders, *_ = raster_fn(
            means=means, quats=quats, scales=scales, opacities=opacs,
            colors=z_depth, viewmats=viewmat[:1], Ks=Ks,
            width=renderer.image_width, height=renderer.image_height,
            near_plane=renderer.near_plane, far_plane=renderer.far_plane,
            backgrounds=bg)
    else:
        renders, *_ = raster_fn(
            means=means, quats=quats, scales=scales, opacities=opacs,
            colors=z_depth, viewmats=viewmat[:1], Ks=Ks,
            width=renderer.image_width, height=renderer.image_height,
            near_plane=renderer.near_plane, far_plane=renderer.far_plane,
            backgrounds=bg)

    return renders[0, :, :, 0]  # [fH, fW]


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
    _, S_all, Vt_all = np.linalg.svd(centered, full_matrices=False)
    basis = Vt_all[:n_components]
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


def cosine_map_to_heatmap(cos_map):
    """Convert [H,W] cosine similarity → colorized [H,W,3] RGB."""
    cos_clipped = np.clip(cos_map, 0, 1)
    heatmap = cv2.applyColorMap((cos_clipped * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def depth_to_colormap(depth, vmin=None, vmax=None):
    """Convert depth map to colorized visualization (INFERNO-like)."""
    d = depth.copy().astype(np.float32)
    if vmin is None:
        vmin = d[d > 0.01].min() if (d > 0.01).any() else 0
    if vmax is None:
        vmax = d.max()
    d = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
    colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def depth_error_map(pred, gt, max_err=0.5):
    """Compute absolute error map and colorize (red=high error)."""
    valid = gt > 0.01
    err = np.abs(pred - gt)
    err[~valid] = 0
    err_norm = np.clip(err / max_err, 0, 1)
    colored = cv2.applyColorMap((err_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    colored[~valid] = 0
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


def upscale(img, scale, interp=cv2.INTER_LINEAR):
    """Upscale image by integer factor."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=interp)


def add_text(img, text, pos=(5, 20), font_scale=0.5, color=(255, 255, 255),
             thickness=1, bg_color=(0, 0, 0)):
    """Add text with background to image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    result = img.copy()
    cv2.rectangle(result, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2),
                  bg_color, -1)
    cv2.putText(result, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
    return result


def hconcat_with_border(imgs, border=2, border_color=(255, 255, 255)):
    """Horizontally concatenate images with border."""
    parts = []
    for i, img in enumerate(imgs):
        if i > 0:
            h = img.shape[0]
            parts.append(np.full((h, border, 3), border_color, dtype=np.uint8))
        parts.append(img)
    return np.concatenate(parts, axis=1)


def vconcat_with_border(imgs, border=2, border_color=(255, 255, 255)):
    """Vertically concatenate images with border."""
    parts = []
    for i, img in enumerate(imgs):
        if i > 0:
            w = img.shape[1]
            parts.append(np.full((border, w, 3), border_color, dtype=np.uint8))
        parts.append(img)
    return np.concatenate(parts, axis=0)


def make_header(texts, cell_width, height=30, font_scale=0.5, border=2):
    """Create a header row with column labels."""
    cells = []
    for i, text in enumerate(texts):
        cell = np.zeros((height, cell_width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
        x = max(0, (cell_width - tw) // 2)
        y = (height + th) // 2
        cv2.putText(cell, text, (x, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        if i > 0:
            cells.append(np.zeros((height, border, 3), dtype=np.uint8))
        cells.append(cell)
    return np.concatenate(cells, axis=1)


# ── Probe training for depth / segmentation ───────────────────────────────────

def train_depth_probe(features, depth_dir, indices, fH, fW):
    """Train a linear depth probe, return probe model."""
    train_X, train_Y = [], []
    for feat, i in zip(features, indices):
        dpath = Path(depth_dir) / f"depth_{i}.png"
        if not dpath.exists():
            continue
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        d = torch.from_numpy(d.astype(np.float32) / 1000.0)
        d = F.interpolate(d[None, None], (fH, fW), mode="bilinear", align_corners=False).squeeze()
        C = feat.shape[0]
        if feat.shape[1:] != (fH, fW):
            feat = F.interpolate(feat[None], (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
        valid = d > 0.01
        if valid.sum() < 10:
            continue
        train_X.append(feat.reshape(C, -1).T[valid.reshape(-1)])
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


def train_seg_probe(features, sem_dir, indices, fH, fW):
    """Train a linear segmentation probe, return probe and n_classes."""
    train_X, train_Y = [], []
    for feat, i in zip(features, indices):
        spath = Path(sem_dir) / f"semantic_class_{i}.png"
        if not spath.exists():
            continue
        sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        sem = torch.from_numpy(sem.astype(np.int64))
        sem = F.interpolate(sem.float()[None, None], (fH, fW), mode="nearest").squeeze().long()
        C = feat.shape[0]
        if feat.shape[1:] != (fH, fW):
            feat = F.interpolate(feat[None], (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
        train_X.append(feat.reshape(C, -1).T)
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


def predict_depth(probe, feat, fH, fW):
    """Predict depth from feature using probe."""
    C = feat.shape[0]
    if feat.shape[1:] != (fH, fW):
        feat = F.interpolate(feat[None].to(device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
    else:
        feat = feat.to(device)
    with torch.no_grad():
        pred = probe(feat.reshape(C, -1).T).squeeze().reshape(fH, fW)
    return pred.cpu().numpy()


def predict_seg(probe, feat, fH, fW):
    """Predict segmentation from feature using probe."""
    C = feat.shape[0]
    if feat.shape[1:] != (fH, fW):
        feat = F.interpolate(feat[None].to(device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
    else:
        feat = feat.to(device)
    with torch.no_grad():
        pred = probe(feat.reshape(C, -1).T).argmax(1).reshape(fH, fW)
    return pred.cpu().numpy()


# ── SigLIP2 grounding ────────────────────────────────────────────────────────

def load_siglip2_projection(projection_weights):
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


def compute_grounding_heatmaps(features_1280, proj_model, text_emb, temperature=1.0):
    """Compute text grounding heatmaps with softmax normalization.

    Args:
        features_1280: [1, 1280, H, W]
        proj_model: SigLIP2 projection
        text_emb: [K, 1536] normalized text embeddings
        temperature: softmax temperature for cross-query normalization

    Returns:
        raw_sim: [K, H, W] raw cosine similarity heatmaps
        softmax_probs: [K, H, W] softmax-normalized probabilities across queries
    """
    B, C, H, W = features_1280.shape
    feat_flat = features_1280.reshape(B, C, H * W).permute(0, 2, 1)
    with torch.no_grad():
        siglip = proj_model(feat_flat.half())
    siglip = F.normalize(siglip, dim=-1).squeeze(0)  # [HW, 1536]
    raw_sim = text_emb @ siglip.T  # [K, HW]
    raw_sim = raw_sim.float().reshape(-1, H, W)

    # Softmax across queries per pixel for zero-shot segmentation
    sim_flat = raw_sim.reshape(raw_sim.shape[0], -1)  # [K, HW]
    probs = F.softmax(sim_flat / temperature, dim=0)  # softmax across K queries
    probs = probs.reshape(raw_sim.shape)  # [K, H, W]

    return raw_sim, probs


# ── Main visualization ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RADIO-GS Comprehensive Visualization")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="/root/results")
    parser.add_argument("--num_views", type=int, default=10,
                        help="Number of novel-view frames to visualize")
    parser.add_argument("--n_train", type=int, default=200,
                        help="Number of training frames for probes")
    parser.add_argument("--scale", type=int, default=16,
                        help="Upscale factor for feature-resolution images")
    parser.add_argument("--text_embeddings",
                        default="output/radio_gs/siglip2_text_embeddings_v2.pt")
    parser.add_argument("--projection_weights",
                        default="output/radio_gs/siglip2_feat_projection.pth")
    parser.add_argument("--grounding_queries", nargs="+",
                        default=["chair", "table", "sofa", "plant", "shelf",
                                 "cushion", "floor", "wall", "door", "window"])
    args = parser.parse_args()

    S = args.scale
    out_root = Path(args.output_dir)

    # Create output subdirectories
    dirs = {}
    for name in ["feature_pca", "depth", "segmentation", "grounding", "composite"]:
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d

    # Load pipeline
    print("Loading RADIO-GS pipeline...")
    model, codec, renderer, sharpener, refiner, config = load_pipeline(
        args.config, args.checkpoint)

    scene = getattr(config, "scene", "room_0")
    scene_root = Path("dataset") / scene
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")
    fH = getattr(config, "feature_height", 30)
    fW = getattr(config, "feature_width", 40)

    # Load val poses
    val_pose_file = scene_root / val_split / "traj_w_c.txt"
    all_c2w = np.loadtxt(str(val_pose_file)).reshape(-1, 4, 4).astype(np.float32)
    all_w2c = np.linalg.inv(all_c2w)
    n_total = len(all_w2c)

    gt_feat_dir = Path(f"output/radio_features_1280d/{scene}/{val_split}/backbone")
    rgb_dir = scene_root / val_split / "rgb"
    depth_dir = scene_root / val_split / "depth"
    sem_dir = scene_root / val_split / "semantic_class"

    # Select evenly-spaced novel-view frames
    vis_indices = list(range(0, n_total, max(1, n_total // args.num_views)))[:args.num_views]
    print(f"Visualizing {len(vis_indices)} novel views from {val_split}")

    # ── Step 1: Render all features ──────────────────────────────────────────
    print("\n[1/6] Rendering decoded features for visualization frames...")
    gt_feats, rend_feats, geom_depths = [], [], []
    with torch.no_grad():
        for i in tqdm(vis_indices, desc="Rendering"):
            gt = torch.load(gt_feat_dir / f"rgb_{i}.pt", map_location="cpu").float()
            if gt.dim() == 2:
                gt = gt.reshape(fH, fW, -1).permute(2, 0, 1)
            gt_feats.append(gt)

            pose = torch.from_numpy(all_w2c[i:i + 1]).to(device)
            decoded, geom_depth, alpha_map = render_features(
                model, codec, renderer, sharpener, refiner, config, pose)
            rend_feats.append(decoded.squeeze(0).cpu())
            if geom_depth is not None:
                geom_depths.append(geom_depth.cpu().numpy())
            else:
                geom_depths.append(None)

    # ── Step 2: Train probes on training data ────────────────────────────────
    print("\n[2/6] Training linear probes on training split features...")
    train_pose_file = scene_root / train_split / "traj_w_c.txt"
    train_c2w = np.loadtxt(str(train_pose_file)).reshape(-1, 4, 4).astype(np.float32)
    train_w2c = np.linalg.inv(train_c2w)
    n_train_total = len(train_w2c)
    train_indices = list(range(0, n_train_total, max(1, n_train_total // args.n_train)))[:args.n_train]

    # Render training features
    train_gt_feats, train_rend_feats = [], []
    gt_train_dir = Path(f"output/radio_features_1280d/{scene}/{train_split}/backbone")
    train_depth_dir = scene_root / train_split / "depth"
    train_sem_dir = scene_root / train_split / "semantic_class"

    print("  Rendering training features...")
    with torch.no_grad():
        for i in tqdm(train_indices, desc="Train render", leave=False):
            gt = torch.load(gt_train_dir / f"rgb_{i}.pt", map_location="cpu").float()
            if gt.dim() == 2:
                gt = gt.reshape(fH, fW, -1).permute(2, 0, 1)
            train_gt_feats.append(gt)

            pose = torch.from_numpy(train_w2c[i:i + 1]).to(device)
            decoded, _, _ = render_features(model, codec, renderer, sharpener, refiner,
                                      config, pose)
            train_rend_feats.append(decoded.squeeze(0).cpu())

    # Train oracle probes (on GT features)
    print("  Training oracle depth probe...")
    oracle_depth_probe = train_depth_probe(train_gt_feats, train_depth_dir,
                                           train_indices, fH, fW)
    print("  Training oracle segmentation probe...")
    oracle_seg_probe, n_classes = train_seg_probe(train_gt_feats, train_sem_dir,
                                                  train_indices, fH, fW)
    # Train rendered probes
    print("  Training rendered depth probe...")
    rend_depth_probe = train_depth_probe(train_rend_feats, train_depth_dir,
                                         train_indices, fH, fW)
    print("  Training rendered segmentation probe...")
    rend_seg_probe, _ = train_seg_probe(train_rend_feats, train_sem_dir,
                                        train_indices, fH, fW)

    # ── Step 3: Feature PCA visualization ────────────────────────────────────
    print("\n[3/6] Generating feature PCA visualizations...")
    all_for_pca = gt_feats + rend_feats
    pca_imgs = shared_pca_colorize(all_for_pca, n_components=3)
    gt_pcas = pca_imgs[:len(gt_feats)]
    rend_pcas = pca_imgs[len(gt_feats):]

    for j, idx in enumerate(vis_indices):
        gt_f, rend_f = gt_feats[j], rend_feats[j]
        cos = F.cosine_similarity(
            rend_f.reshape(-1, fH * fW), gt_f.reshape(-1, fH * fW), dim=0
        ).reshape(fH, fW).numpy()

        # Load GT RGB
        rgb = cv2.imread(str(rgb_dir / f"rgb_{idx}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (fW, fH))

        panels = [
            upscale(rgb_small, S, cv2.INTER_LINEAR),
            upscale(gt_pcas[j], S),
            upscale(rend_pcas[j], S),
            upscale(cosine_map_to_heatmap(cos), S),
        ]

        labels = ["Input RGB", "GT Feature (PCA)", "Rendered Feature (PCA)",
                   f"Cosine Sim (mean={cos.mean():.3f})"]
        for k, label in enumerate(labels):
            panels[k] = add_text(panels[k], label, pos=(5, 20), font_scale=0.55)

        row = hconcat_with_border(panels, border=3)
        save_path = dirs["feature_pca"] / f"pca_frame_{idx:04d}_cos{cos.mean():.3f}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # PCA summary grid (up to 8 frames)
    n_grid = min(8, len(vis_indices))
    grid_rows = []
    for j in range(n_grid):
        idx = vis_indices[j]
        cos = F.cosine_similarity(
            rend_feats[j].reshape(-1, fH * fW), gt_feats[j].reshape(-1, fH * fW), dim=0
        ).reshape(fH, fW).numpy()
        rgb = cv2.imread(str(rgb_dir / f"rgb_{idx}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        panels = [
            upscale(cv2.resize(rgb, (fW, fH)), S, cv2.INTER_LINEAR),
            upscale(gt_pcas[j], S),
            upscale(rend_pcas[j], S),
            upscale(cosine_map_to_heatmap(cos), S),
        ]
        grid_rows.append(hconcat_with_border(panels, border=2))

    header = make_header(["Input RGB", "GT Feature (PCA)", "Rendered (PCA)", "Cosine Similarity"],
                         fW * S, height=30, border=2)
    full_grid = vconcat_with_border([header] + grid_rows, border=2)
    cv2.imwrite(str(dirs["feature_pca"] / "pca_grid.png"),
                cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved PCA grid and {len(vis_indices)} individual frames")

    # ── Step 4: Depth visualization ──────────────────────────────────────────
    print("\n[4/6] Generating depth estimation visualizations...")
    for j, idx in enumerate(vis_indices):
        # GT depth
        dpath = depth_dir / f"depth_{idx}.png"
        d_raw = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d_raw is None:
            continue
        gt_depth = d_raw.astype(np.float32) / 1000.0
        gt_depth_feat = cv2.resize(gt_depth, (fW, fH), interpolation=cv2.INTER_LINEAR)

        vmin = gt_depth_feat[gt_depth_feat > 0.01].min() if (gt_depth_feat > 0.01).any() else 0
        vmax = gt_depth_feat.max()

        # Oracle prediction
        oracle_pred = predict_depth(oracle_depth_probe, gt_feats[j], fH, fW)
        # Rendered prediction
        rend_pred = predict_depth(rend_depth_probe, rend_feats[j], fH, fW)

        # Geometry depth from 3DGS
        geom_d = geom_depths[j]
        if geom_d is not None:
            if geom_d.ndim == 2 and geom_d.shape == (fH, fW):
                geom_depth_vis = geom_d
            else:
                geom_depth_vis = cv2.resize(geom_d.squeeze(), (fW, fH),
                                            interpolation=cv2.INTER_LINEAR)
        else:
            geom_depth_vis = np.zeros((fH, fW), dtype=np.float32)

        rgb = cv2.imread(str(rgb_dir / f"rgb_{idx}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (fW, fH))

        panels = [
            upscale(rgb_small, S, cv2.INTER_LINEAR),
            upscale(depth_to_colormap(gt_depth_feat, vmin, vmax), S),
            upscale(depth_to_colormap(geom_depth_vis, vmin, vmax), S),
            upscale(depth_to_colormap(oracle_pred, vmin, vmax), S),
            upscale(depth_to_colormap(rend_pred, vmin, vmax), S),
            upscale(depth_error_map(rend_pred, gt_depth_feat), S),
        ]

        labels = ["Input RGB", "GT Depth", "Geom Depth", "Oracle Pred",
                   "Rendered Pred", "Rendered Error"]
        for k, label in enumerate(labels):
            panels[k] = add_text(panels[k], label, pos=(5, 20), font_scale=0.5)

        row = hconcat_with_border(panels, border=3)
        save_path = dirs["depth"] / f"depth_frame_{idx:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # Depth grid
    grid_rows = []
    for j in range(min(8, len(vis_indices))):
        idx = vis_indices[j]
        dpath = depth_dir / f"depth_{idx}.png"
        d_raw = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d_raw is None:
            continue
        gt_d = d_raw.astype(np.float32) / 1000.0
        gt_d_f = cv2.resize(gt_d, (fW, fH), interpolation=cv2.INTER_LINEAR)
        vmin = gt_d_f[gt_d_f > 0.01].min() if (gt_d_f > 0.01).any() else 0
        vmax = gt_d_f.max()
        r_pred = predict_depth(rend_depth_probe, rend_feats[j], fH, fW)
        geom_d = geom_depths[j]
        if geom_d is not None:
            if geom_d.ndim == 2 and geom_d.shape == (fH, fW):
                geom_d_vis = geom_d
            else:
                geom_d_vis = cv2.resize(geom_d.squeeze(), (fW, fH),
                                        interpolation=cv2.INTER_LINEAR)
        else:
            geom_d_vis = np.zeros((fH, fW), dtype=np.float32)
        rgb = cv2.imread(str(rgb_dir / f"rgb_{idx}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        panels = [
            upscale(cv2.resize(rgb, (fW, fH)), S, cv2.INTER_LINEAR),
            upscale(depth_to_colormap(gt_d_f, vmin, vmax), S),
            upscale(depth_to_colormap(geom_d_vis, vmin, vmax), S),
            upscale(depth_to_colormap(r_pred, vmin, vmax), S),
            upscale(depth_error_map(r_pred, gt_d_f), S),
        ]
        grid_rows.append(hconcat_with_border(panels, border=2))

    if grid_rows:
        header = make_header(["Input RGB", "GT Depth", "Geom Depth", "Rendered Pred", "Error Map"],
                             fW * S, height=30, border=2)
        full_grid = vconcat_with_border([header] + grid_rows, border=2)
        cv2.imwrite(str(dirs["depth"] / "depth_grid.png"),
                    cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved depth grid and {len(vis_indices)} individual frames")

    # ── Step 5: Segmentation visualization ───────────────────────────────────
    print("\n[5/6] Generating segmentation visualizations...")
    for j, idx in enumerate(vis_indices):
        spath = sem_dir / f"semantic_class_{idx}.png"
        sem_raw = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
        if sem_raw is None:
            continue
        gt_sem = cv2.resize(sem_raw, (fW, fH), interpolation=cv2.INTER_NEAREST)

        oracle_seg = predict_seg(oracle_seg_probe, gt_feats[j], fH, fW)
        rend_seg = predict_seg(rend_seg_probe, rend_feats[j], fH, fW)

        rgb = cv2.imread(str(rgb_dir / f"rgb_{idx}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (fW, fH))

        # Blend seg overlay with RGB
        alpha = 0.6
        gt_seg_rgb = seg_to_color(gt_sem)
        oracle_seg_rgb = seg_to_color(oracle_seg)
        rend_seg_rgb = seg_to_color(rend_seg)

        gt_blend = (alpha * gt_seg_rgb + (1 - alpha) * rgb_small).astype(np.uint8)
        oracle_blend = (alpha * oracle_seg_rgb + (1 - alpha) * rgb_small).astype(np.uint8)
        rend_blend = (alpha * rend_seg_rgb + (1 - alpha) * rgb_small).astype(np.uint8)

        panels = [
            upscale(rgb_small, S, cv2.INTER_LINEAR),
            upscale(gt_blend, S, cv2.INTER_NEAREST),
            upscale(oracle_blend, S, cv2.INTER_NEAREST),
            upscale(rend_blend, S, cv2.INTER_NEAREST),
        ]

        labels = ["Input RGB", "GT Segmentation", "Oracle Pred", "Rendered Pred"]
        for k, label in enumerate(labels):
            panels[k] = add_text(panels[k], label, pos=(5, 20), font_scale=0.5)

        row = hconcat_with_border(panels, border=3)
        save_path = dirs["segmentation"] / f"seg_frame_{idx:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # Segmentation grid
    grid_rows = []
    for j in range(min(8, len(vis_indices))):
        idx = vis_indices[j]
        spath = sem_dir / f"semantic_class_{idx}.png"
        sem_raw = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
        if sem_raw is None:
            continue
        gt_sem = cv2.resize(sem_raw, (fW, fH), interpolation=cv2.INTER_NEAREST)
        r_seg = predict_seg(rend_seg_probe, rend_feats[j], fH, fW)
        rgb = cv2.imread(str(rgb_dir / f"rgb_{idx}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_s = cv2.resize(rgb, (fW, fH))
        gt_blend = (0.6 * seg_to_color(gt_sem) + 0.4 * rgb_s).astype(np.uint8)
        rend_blend = (0.6 * seg_to_color(r_seg) + 0.4 * rgb_s).astype(np.uint8)
        panels = [
            upscale(rgb_s, S, cv2.INTER_LINEAR),
            upscale(gt_blend, S, cv2.INTER_NEAREST),
            upscale(rend_blend, S, cv2.INTER_NEAREST),
        ]
        grid_rows.append(hconcat_with_border(panels, border=2))

    if grid_rows:
        header = make_header(["Input RGB", "GT Segmentation", "Rendered Pred"],
                             fW * S, height=30, border=2)
        full_grid = vconcat_with_border([header] + grid_rows, border=2)
        cv2.imwrite(str(dirs["segmentation"] / "seg_grid.png"),
                    cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved segmentation grid and {len(vis_indices)} individual frames")

    # ── Step 6: Text grounding heatmaps ──────────────────────────────────────
    print("\n[6/6] Generating text grounding heatmaps...")
    text_emb_path = Path(args.text_embeddings)
    proj_path = Path(args.projection_weights)
    # Create grounding_seg output dir
    grounding_seg_dir = out_root / "grounding_seg"
    grounding_seg_dir.mkdir(parents=True, exist_ok=True)

    if text_emb_path.exists() and proj_path.exists():
        text_data = torch.load(str(text_emb_path), map_location="cpu")
        all_queries = text_data["queries"]
        all_text_emb = text_data["embeddings"].to(device).half()
        query_to_idx = {q: i for i, q in enumerate(all_queries)}

        proj_model = load_siglip2_projection(str(proj_path))

        # Filter to requested queries
        active_queries = [q for q in args.grounding_queries if q in query_to_idx]
        active_text_emb = torch.stack([all_text_emb[query_to_idx[q]] for q in active_queries])
        # Also get class IDs for mask overlay
        name_to_cid = {v: k for k, v in REPLICA_CLASSES.items()}

        # Assign colors for grounding queries
        np.random.seed(123)
        query_colors = {q: tuple(np.random.randint(80, 255, 3).tolist())
                        for q in active_queries}

        print(f"  Active grounding queries: {active_queries}")

        for j, idx in enumerate(vis_indices):
            gt_f = gt_feats[j].unsqueeze(0).to(device)
            rend_f = rend_feats[j].unsqueeze(0).to(device)

            gt_raw, gt_probs = compute_grounding_heatmaps(gt_f, proj_model, active_text_emb)
            rend_raw, rend_probs = compute_grounding_heatmaps(rend_f, proj_model, active_text_emb)

            # Load semantic GT for mask overlay
            spath = sem_dir / f"semantic_class_{idx}.png"
            sem_raw = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
            if sem_raw is not None:
                gt_sem = cv2.resize(sem_raw, (fW, fH), interpolation=cv2.INTER_NEAREST)
            else:
                gt_sem = None

            rgb = cv2.imread(str(rgb_dir / f"rgb_{idx}.png"))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (fW, fH))

            # Per-query rows: Query Label | GT Mask | GT Heatmap | Rendered Heatmap
            query_rows = []
            for qi, qname in enumerate(active_queries):
                gt_h = gt_raw[qi].cpu().numpy()
                rend_h = rend_raw[qi].cpu().numpy()

                # Per-query min-max normalization (independent for GT and rendered)
                def per_query_norm(h):
                    lo, hi = h.min(), h.max()
                    if hi - lo > 1e-6:
                        return (h - lo) / (hi - lo)
                    return np.zeros_like(h)

                gt_norm = per_query_norm(gt_h)
                rend_norm = per_query_norm(rend_h)

                gt_color = cv2.applyColorMap((gt_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                gt_color = cv2.cvtColor(gt_color, cv2.COLOR_BGR2RGB)
                rend_color = cv2.applyColorMap((rend_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                rend_color = cv2.cvtColor(rend_color, cv2.COLOR_BGR2RGB)

                # GT semantic mask for this class
                if gt_sem is not None and qname in name_to_cid:
                    cid = name_to_cid[qname]
                    mask = (gt_sem == cid).astype(np.uint8)
                    mask_vis = np.zeros((fH, fW, 3), dtype=np.uint8)
                    mask_vis[mask > 0] = (0, 255, 100)
                    mask_blend = (0.5 * mask_vis + 0.5 * rgb_small).astype(np.uint8)
                else:
                    mask_blend = np.zeros_like(rgb_small)

                # Heatmap overlays on RGB
                gt_overlay = (0.5 * gt_color + 0.5 * rgb_small).astype(np.uint8)
                rend_overlay = (0.5 * rend_color + 0.5 * rgb_small).astype(np.uint8)

                panels = [
                    upscale(mask_blend, S, cv2.INTER_NEAREST),
                    upscale(gt_overlay, S),
                    upscale(rend_overlay, S),
                ]
                # Add query label
                panels[0] = add_text(panels[0], qname, pos=(5, 20), font_scale=0.55)
                query_rows.append(hconcat_with_border(panels, border=2))

            if query_rows:
                header = make_header(["GT Mask", "GT Heatmap", "Rendered Heatmap"],
                                     fW * S, height=28, border=2)
                # Add RGB panel at top
                rgb_up = upscale(rgb_small, S, cv2.INTER_LINEAR)
                rgb_labeled = add_text(rgb_up, f"Frame {idx}", pos=(5, 20), font_scale=0.55)
                rgb_row = np.zeros((rgb_up.shape[0], header.shape[1], 3), dtype=np.uint8)
                x_off = (rgb_row.shape[1] - rgb_up.shape[1]) // 2
                rgb_row[:, x_off:x_off + rgb_up.shape[1]] = rgb_labeled

                full = vconcat_with_border([rgb_row, header] + query_rows, border=2)
                save_path = dirs["grounding"] / f"grounding_frame_{idx:04d}.png"
                cv2.imwrite(str(save_path), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))

            # ── Zero-shot segmentation from softmax grounding ──
            rend_seg_argmax = rend_probs.argmax(dim=0).cpu().numpy()  # [H, W]
            rend_seg_color = np.zeros((fH, fW, 3), dtype=np.uint8)
            for qi, qname in enumerate(active_queries):
                mask = rend_seg_argmax == qi
                if mask.any():
                    rend_seg_color[mask] = query_colors[qname]

            gt_seg_argmax = gt_probs.argmax(dim=0).cpu().numpy()
            gt_seg_color = np.zeros((fH, fW, 3), dtype=np.uint8)
            for qi, qname in enumerate(active_queries):
                mask = gt_seg_argmax == qi
                if mask.any():
                    gt_seg_color[mask] = query_colors[qname]

            panels = [
                upscale(rgb_small, S, cv2.INTER_LINEAR),
                upscale((0.5 * gt_seg_color + 0.5 * rgb_small).astype(np.uint8), S, cv2.INTER_NEAREST),
                upscale((0.5 * rend_seg_color + 0.5 * rgb_small).astype(np.uint8), S, cv2.INTER_NEAREST),
            ]
            labels_seg = ["Input RGB", "GT Softmax Seg", "Rendered Softmax Seg"]
            for k, label in enumerate(labels_seg):
                panels[k] = add_text(panels[k], label, pos=(5, 20), font_scale=0.5)
            row = hconcat_with_border(panels, border=3)
            cv2.imwrite(str(grounding_seg_dir / f"grounding_seg_{idx:04d}.png"),
                        cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

        print(f"  Saved grounding visualizations for {len(vis_indices)} frames")
    else:
        print("  ⚠ Skipping grounding: text embeddings or projection weights not found")

    # ── Step 7: Composite multi-task figure ──────────────────────────────────
    print("\nGenerating composite multi-task figures...")
    for j, idx in enumerate(vis_indices[:5]):  # Top 5 frames
        rgb = cv2.imread(str(rgb_dir / f"rgb_{idx}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (fW, fH))

        # PCA
        cos = F.cosine_similarity(
            rend_feats[j].reshape(-1, fH * fW), gt_feats[j].reshape(-1, fH * fW), dim=0
        ).reshape(fH, fW).numpy()
        pca_panel = upscale(rend_pcas[j], S)
        cos_panel = upscale(cosine_map_to_heatmap(cos), S)

        # Depth
        dpath = depth_dir / f"depth_{idx}.png"
        d_raw = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d_raw is not None:
            gt_d = d_raw.astype(np.float32) / 1000.0
            gt_d_f = cv2.resize(gt_d, (fW, fH), interpolation=cv2.INTER_LINEAR)
            vmin_d = gt_d_f[gt_d_f > 0.01].min() if (gt_d_f > 0.01).any() else 0
            vmax_d = gt_d_f.max()
            r_depth = predict_depth(rend_depth_probe, rend_feats[j], fH, fW)
            gt_depth_panel = upscale(depth_to_colormap(gt_d_f, vmin_d, vmax_d), S)
            rend_depth_panel = upscale(depth_to_colormap(r_depth, vmin_d, vmax_d), S)
        else:
            h_, w_ = fH * S, fW * S
            gt_depth_panel = np.zeros((h_, w_, 3), dtype=np.uint8)
            rend_depth_panel = np.zeros((h_, w_, 3), dtype=np.uint8)

        # Segmentation
        spath = sem_dir / f"semantic_class_{idx}.png"
        sem_raw = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
        if sem_raw is not None:
            gt_sem = cv2.resize(sem_raw, (fW, fH), interpolation=cv2.INTER_NEAREST)
            r_seg = predict_seg(rend_seg_probe, rend_feats[j], fH, fW)
            gt_seg_panel = upscale((0.6 * seg_to_color(gt_sem) + 0.4 * rgb_small).astype(np.uint8), S)
            rend_seg_panel = upscale((0.6 * seg_to_color(r_seg) + 0.4 * rgb_small).astype(np.uint8), S)
        else:
            h_, w_ = fH * S, fW * S
            gt_seg_panel = np.zeros((h_, w_, 3), dtype=np.uint8)
            rend_seg_panel = np.zeros((h_, w_, 3), dtype=np.uint8)

        # Layout: 2 rows × 5 cols
        # Row 1: RGB | Feature PCA | GT Depth | Geom Depth | GT Seg
        # Row 2: Cosine Sim | (blank) | Rendered Depth | Error | Rendered Seg
        cell_w = fW * S
        cell_h = fH * S
        rgb_panel = upscale(rgb_small, S, cv2.INTER_LINEAR)

        # Geometry depth
        geom_d = geom_depths[j]
        if d_raw is not None and geom_d is not None:
            if geom_d.ndim == 2 and geom_d.shape == (fH, fW):
                geom_d_vis = geom_d
            else:
                geom_d_vis = cv2.resize(geom_d.squeeze(), (fW, fH),
                                        interpolation=cv2.INTER_LINEAR)
            geom_depth_panel = upscale(depth_to_colormap(geom_d_vis, vmin_d, vmax_d), S)
        else:
            geom_depth_panel = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

        row1_panels = [rgb_panel, pca_panel, gt_depth_panel, geom_depth_panel, gt_seg_panel]
        row1_labels = ["Input RGB", "Feature PCA", "GT Depth", "Geom Depth", "GT Segmentation"]
        row2_panels = [cos_panel, np.zeros((cell_h, cell_w, 3), dtype=np.uint8),
                       rend_depth_panel,
                       upscale(depth_error_map(r_depth, gt_d_f), S) if d_raw is not None else np.zeros((cell_h, cell_w, 3), dtype=np.uint8),
                       rend_seg_panel]
        row2_labels = [f"Cosine ({cos.mean():.3f})", "", "Pred Depth", "Depth Error", "Pred Seg"]

        for k in range(5):
            row1_panels[k] = add_text(row1_panels[k], row1_labels[k], font_scale=0.45)
            if row2_labels[k]:
                row2_panels[k] = add_text(row2_panels[k], row2_labels[k], font_scale=0.45)

        row1 = hconcat_with_border(row1_panels, border=3)
        row2 = hconcat_with_border(row2_panels, border=3)
        composite = vconcat_with_border([row1, row2], border=3)

        save_path = dirs["composite"] / f"composite_frame_{idx:04d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    print(f"  Saved {min(5, len(vis_indices))} composite figures")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {out_root}")
    for name, d in dirs.items():
        n_files = len(list(d.glob("*.png")))
        print(f"  {name}/: {n_files} images")

    # Compute and print quality stats
    cos_vals = []
    for j in range(len(vis_indices)):
        cos = F.cosine_similarity(
            rend_feats[j].flatten().unsqueeze(0),
            gt_feats[j].flatten().unsqueeze(0)
        ).item()
        cos_vals.append(cos)
    print(f"\nFeature quality (val novel views, {len(vis_indices)} frames):")
    print(f"  Mean cosine similarity: {np.mean(cos_vals):.4f}")
    print(f"  Min cosine similarity:  {np.min(cos_vals):.4f}")


if __name__ == "__main__":
    main()
