"""Generate improved visualizations for RADIO-GS qualitative evaluation (v2).

Fixes over v1 (generate_visualizations.py):
  1. Grounding: per-query min-max normalization + softmax across queries for
     zero-shot segmentation.  Adds GT semantic mask column.
  2. Depth: shows 3DGS geometry depth alongside feature-predicted depth.
  3. Resolution: bilinear upscaling (cv2.INTER_LINEAR) for heatmaps/PCA/depth,
     INTER_NEAREST only for class-label masks.  Default scale 16 → 480×640.

Usage:
    python radio_gs/scripts/generate_visualizations_v2.py \
        --config radio_gs/configs/replica_explicit_v11.yaml \
        --checkpoint output/radio_gs/room0_explicit_v11/checkpoints/best.pth \
        --output_dir /root/results/v2 \
        --num_views 10
"""
from __future__ import annotations

import argparse
import gc
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
from radio_gs.models.depth_fusion import (
    predict_depth_fusion,
    prepare_depth_fusion_sample,
    sample_depth_fusion_training_pixels,
    train_depth_fusion_probe,
)
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.screen_refiner import (
    ScreenSpaceRefiner,
    build_depth_guide,
    build_refiner_guide,
    compute_refiner_extra_channels,
)
from radio_gs.data.benchmark_paths import resolve_split_feature_dir
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

device = torch.device("cuda")
probe_device = torch.device("cpu")
DEFAULT_SIGLIP2_TEXT_EMBEDDINGS = "output/radio_gs/siglip2_text_embeddings_v2.pt"

def inference_dtype(target_device: torch.device) -> torch.dtype:
    return torch.float16 if target_device.type == "cuda" else torch.float32


# Replica semantic classes
from radio_gs.replica_constants import (
    REPLICA_CLASSES, GROUNDING_QUERIES as GROUNDING_QUERY_CLASS_IDS,
    SEG_COLORS,
)


# ── Pipeline loading ──────────────────────────────────────────────────────────

def build_renderer(config, image_height=None, image_width=None):
    """Build a feature/RGB renderer at the requested output resolution."""
    H = image_height or getattr(config, "feature_height", 30)
    W = image_width or getattr(config, "feature_width", 40)
    return FeatureFieldRenderer(
        image_height=H, image_width=W,
        fx=getattr(config, "fx", 320.0) * W / getattr(config, "image_width", 640),
        fy=getattr(config, "fy", 320.0) * H / getattr(config, "image_height", 480),
        cx=getattr(config, "cx", 319.5) * W / getattr(config, "image_width", 640),
        cy=getattr(config, "cy", 239.5) * H / getattr(config, "image_height", 480),
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=getattr(config, "use_2dgs", False),
    ).to(device)


def load_pipeline(config_path, checkpoint_path):
    """Load the full RADIO-GS rendering pipeline."""
    config = load_config(config_path)

    architecture = getattr(config, "architecture", "explicit")
    is_hybrid = architecture == "hybrid"

    if is_hybrid:
        from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
        latent_dim = getattr(config, "hybrid_latent_dim", 16)
        model = HybridFeatureGaussian(
            latent_dim=latent_dim,
            hash_output_dim=getattr(config, "hash_output_dim", 48),
            fine_dim=getattr(config, "fine_dim", 64),
            coarse_dim=getattr(config, "coarse_dim", 64),
            output_dim=getattr(config, "hybrid_output_dim", 128),
            num_levels=getattr(config, "hash_levels", 16),
            features_per_level=getattr(config, "hash_features_per_level", 2),
            log2_hashmap_size=getattr(config, "hash_log2_size", 19),
            base_resolution=getattr(config, "hash_base_resolution", 16),
            max_resolution=getattr(config, "hash_max_resolution", 2048),
            decoupled_heads=getattr(config, "hybrid_decoupled_heads", False),
            use_semantic_adaptor=getattr(config, "hybrid_semantic_adaptor", False),
            semantic_adaptor_mode=getattr(config, "hybrid_semantic_adaptor_mode", "confidence"),
            semantic_adaptor_hidden_dim=getattr(config, "hybrid_semantic_adaptor_hidden_dim", 64),
            semantic_adaptor_use_geometry_guidance=getattr(
                config, "hybrid_semantic_adaptor_use_geometry_guidance", True
            ),
            semantic_adaptor_use_depth_guidance=getattr(
                config, "hybrid_semantic_adaptor_use_depth_guidance", False
            ),
            semantic_adaptor_residual=getattr(
                config, "hybrid_semantic_adaptor_residual", True
            ),
        )
    else:
        latent_dim = getattr(config, "latent_dim", 64)
        model = ExplicitFeatureGaussian(latent_dim=latent_dim)

    ply_path = getattr(config, "ply_path", "")
    if ply_path:
        model.load_from_ply(ply_path)
    model = model.to(device).eval()

    codec = HCDCodec(
        input_dim=getattr(config, "radio_feature_dim", 1280),
        bottleneck_dim=getattr(config, "bottleneck_dim", 64),
        dual_stream=getattr(config, "dual_stream", True),
        symmetric_decoder=getattr(config, "symmetric_decoder", False),
    ).to(device).eval()

    renderer = build_renderer(config)

    sharpener = FeatSharp3D(
        mode=getattr(config, "featsharp_mode", "analytical"),
        feature_dim=latent_dim,
        strength=getattr(config, "featsharp_strength", 0.3),
    ).to(device).eval()

    refiner = None
    rgb_guide_enabled = getattr(config, "refiner_rgb_guide", False)
    depth_guide_enabled = getattr(config, "refiner_depth_guide", False)
    depth_grad_enabled = getattr(config, "refiner_depth_grad", False)
    if getattr(config, "use_refiner", False):
        extra_ch = compute_refiner_extra_channels(
            rgb_guide=rgb_guide_enabled,
            depth_guide=depth_guide_enabled,
            depth_grad=depth_grad_enabled,
            alpha_guide=getattr(config, "refiner_alpha_guide", False),
            boundary_guide=getattr(config, "refiner_boundary_guide", False),
        )
        norm_type = getattr(config, "refiner_norm_type", "gn")
        refiner = ScreenSpaceRefiner(
            latent_dim=latent_dim,
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

    return model, codec, renderer, sharpener, refiner, config, is_hybrid


def _hybrid_decode(model, rendered, result, pose_w2c, K):
    """Apply hybrid hash-grid decode to rendered features."""
    from radio_gs.models.hybrid_gaussian import unproject_depth_to_positions
    depth_map = result["depth_map"].float()
    H, W = depth_map.shape[1], depth_map.shape[2]
    position_map = unproject_depth_to_positions(depth_map, pose_w2c.float(), K.float(), H, W)
    xyz = model.get_xyz()
    margin = 0.1
    lo = xyz.min(dim=0).values - margin
    hi = xyz.max(dim=0).values + margin
    extent = (hi - lo).clamp(min=1e-6)
    position_map = ((position_map - lo.view(1, 3, 1, 1)) / extent.view(1, 3, 1, 1)).clamp(0, 1)
    return model.decode_screen_space(
        rendered.float(),
        position_map,
        depth_map=depth_map,
    )


def _build_depth_guide(render_result, depth_grad=False, grad_scale=10.0):
    """Backwards-compatible wrapper for shared depth-guide logic."""
    return build_depth_guide(
        render_result["depth_map"],
        depth_grad=depth_grad,
        grad_scale=grad_scale,
    )


def render_features(model, codec, renderer, sharpener, refiner, config, viewmat,
                    is_hybrid=False):
    """Render and decode 1280d features + geometry depth for a single view."""
    self_guided = getattr(config, "self_guided", False)
    depth_guide_enabled = getattr(config, "refiner_depth_guide", False)
    depth_grad_enabled = getattr(config, "refiner_depth_grad", False)
    with torch.no_grad():
        if self_guided:
            vm = viewmat if viewmat.dim() == 3 else viewmat.unsqueeze(0)
            result = renderer.render_features_and_rgb(model, vm)
            latent = result["feature_map"]
            rgb_guide = result["rgb"]
            geom_depth = result.get("geom_depth", None)
            alpha_map = result.get("geom_alpha", None)
        else:
            result = renderer.render_features_batch(model, viewmat)
            latent = result["feature_map"]
            rgb_guide = None
            geom_depth = None
            alpha_map = None

        # Geometry depth from SH-based rendering. When self-guided rendering already
        # rasterized RGB, reuse that pass instead of launching an extra render.
        if geom_depth is None or alpha_map is None:
            vm_2d = viewmat if viewmat.dim() == 2 else viewmat.squeeze(0)
            rgb_result = renderer.render_rgb(model, vm_2d)
            geom_depth = rgb_result["depth"]      # [fH, fW]
            alpha_map = rgb_result["alpha"]       # [fH, fW]
        else:
            if geom_depth.dim() == 3:
                geom_depth = geom_depth.squeeze(0)
            if alpha_map.dim() == 3:
                alpha_map = alpha_map.squeeze(0)

        latent = sharpener(latent)
        if refiner is not None:
            guide = build_refiner_guide(
                result,
                rgb_guide=rgb_guide,
                use_depth_guide=depth_guide_enabled,
                use_depth_grad=depth_grad_enabled,
                depth_grad_scale=getattr(config, "refiner_depth_grad_scale", 10.0),
                use_alpha_guide=getattr(config, "refiner_alpha_guide", False),
                use_boundary_guide=getattr(config, "refiner_boundary_guide", False),
            )
            latent = refiner(latent, guide=guide)
        if is_hybrid:
            latent = _hybrid_decode(model, latent, result, viewmat, renderer.K)
        decoded = codec.decoder(latent)
    return decoded, geom_depth, alpha_map  # [1, 1280, H, W], [fH, fW], [fH, fW]


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


def resize_depth_map(depth, target_hw, interp=cv2.INTER_LINEAR):
    """Resize a depth map to target (H, W) in float32."""
    target_h, target_w = target_hw
    return cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=interp)


def smooth_depth_for_display(depth, valid_mask=None, guide_rgb=None):
    """Make noisy geometry depth easier to interpret for visualization only.
    
    Uses a single light median filter to remove salt-and-pepper noise without
    adding texture artifacts from edge-preserving bilateral filters.
    """
    d = depth.astype(np.float32).copy()
    if valid_mask is None:
        valid_mask = d > 0.01
    else:
        valid_mask = valid_mask.astype(bool)
    if valid_mask.sum() < 10:
        return d
    ksize = 5 if min(d.shape[:2]) >= 60 else 3
    d_smooth = cv2.medianBlur(d, ksize)
    d_smooth[~valid_mask] = 0.0
    return d_smooth


def grounding_mask_from_probs(prob_map, prob_stack, query_idx, base_threshold=None):
    """Create binary mask from softmax probabilities using per-query thresholding.

    Uses argmax winner + 75th-percentile threshold for robust mask generation.
    """
    winner = prob_stack.argmax(axis=0) == query_idx
    # Threshold at 75th percentile of this query's probability distribution
    thresh = float(np.percentile(prob_map, 75.0))
    thresh = max(thresh, 0.01)
    mask = winner & (prob_map >= thresh)
    if not mask.any():
        # Fallback: just use argmax winner above median
        thresh = float(np.median(prob_map))
        mask = winner & (prob_map >= thresh)
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8


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
    """Upscale image by integer factor.  Default changed to bilinear."""
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

def _build_probe(in_dim, out_dim, hidden=256):
    """Build a 2-layer MLP probe for downstream task visualization."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _train_probe(probe, train_X, train_Y, epochs=300, batch_size=16384,
                 lr=1e-3, task="regression", class_weights=None):
    """Train a probe with mini-batch sampling for stable visualization heads."""
    probe = probe.to(probe_device).train()
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = train_X.shape[0]
    for _ in range(epochs):
        idx = torch.randint(0, n, (min(batch_size, n),), device=train_X.device)
        pred = probe(train_X[idx])
        if task == "regression":
            loss = F.l1_loss(pred.squeeze(), train_Y[idx])
        else:
            loss = F.cross_entropy(pred, train_Y[idx], weight=class_weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
    probe.eval()
    return probe

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

    train_X = torch.cat(train_X, 0).to(probe_device)
    train_Y = torch.cat(train_Y, 0).to(probe_device)
    probe = _build_probe(train_X.shape[1], 1)
    return _train_probe(probe, train_X, train_Y, epochs=300, task="regression")


def train_seg_probe(features, sem_dir, indices, fH, fW):
    """Train a segmentation probe, return probe and class-ID mappings."""
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

    train_X = torch.cat(train_X, 0).to(probe_device)
    train_Y = torch.cat(train_Y, 0).to(probe_device)
    unique_classes = torch.unique(train_Y).tolist()
    id_to_contiguous = {c: i for i, c in enumerate(unique_classes)}
    contiguous_to_id = {i: c for c, i in id_to_contiguous.items()}
    train_Y = torch.tensor([id_to_contiguous[y.item()] for y in train_Y],
                           dtype=torch.long, device=probe_device)
    n_classes = len(unique_classes)
    counts = torch.bincount(train_Y, minlength=n_classes).float().clamp(min=1)
    weights = (1.0 / counts)
    weights = (weights / weights.sum() * n_classes).to(probe_device)
    probe = _build_probe(train_X.shape[1], n_classes)
    probe = _train_probe(probe, train_X, train_Y, epochs=500, task="classification",
                         class_weights=weights)
    return probe, id_to_contiguous, contiguous_to_id


def predict_depth(probe, feat, fH, fW):
    """Predict depth from feature using probe."""
    C = feat.shape[0]
    if feat.shape[1:] != (fH, fW):
        feat = F.interpolate(feat[None].to(probe_device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
    else:
        feat = feat.to(probe_device)
    with torch.no_grad():
        pred = probe(feat.reshape(C, -1).T).squeeze().reshape(fH, fW)
    return pred.cpu().numpy()


def _restore_original_seg_ids(seg_map, contiguous_to_id=None):
    """Map contiguous probe outputs back to original semantic IDs if needed."""
    if contiguous_to_id is None:
        return seg_map
    restored = np.zeros_like(seg_map, dtype=np.int64)
    for cont_id, orig_id in contiguous_to_id.items():
        restored[seg_map == cont_id] = orig_id
    return restored


def predict_seg(probe, feat, fH, fW, contiguous_to_id=None):
    """Predict segmentation from feature using probe (low-res, for metrics)."""
    C = feat.shape[0]
    if feat.shape[1:] != (fH, fW):
        feat = F.interpolate(feat[None].to(probe_device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
    else:
        feat = feat.to(probe_device)
    with torch.no_grad():
        pred = probe(feat.reshape(C, -1).T).argmax(1).reshape(fH, fW)
    return _restore_original_seg_ids(pred.cpu().numpy(), contiguous_to_id)


def predict_seg_smooth(probe, feat, fH, fW, target_h, target_w, contiguous_to_id=None):
    """Predict segmentation with bilinear-upscaled logits for smooth boundaries.
    
    Upscales logits (soft class probabilities) BEFORE argmax so that class
    boundaries are interpolated smoothly instead of blocky nearest-neighbor.
    """
    C = feat.shape[0]
    if feat.shape[1:] != (fH, fW):
        feat = F.interpolate(feat[None].to(probe_device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
    else:
        feat = feat.to(probe_device)
    with torch.no_grad():
        logits = probe(feat.reshape(C, -1).T)  # [fH*fW, n_classes]
        n_classes = logits.shape[1]
        logits_map = logits.T.reshape(1, n_classes, fH, fW)
        logits_up = F.interpolate(logits_map, (target_h, target_w),
                                  mode="bilinear", align_corners=False)
        pred = logits_up.squeeze(0).argmax(0)  # [target_h, target_w]
    return _restore_original_seg_ids(pred.cpu().numpy(), contiguous_to_id)


def train_depth_probe_paths(features, depth_paths, fH, fW):
    """Train a linear depth probe using explicit file paths (mixed_split compatible)."""
    train_X, train_Y = [], []
    for feat, dpath in zip(features, depth_paths):
        dpath = Path(dpath)
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

    train_X = torch.cat(train_X, 0).to(probe_device)
    train_Y = torch.cat(train_Y, 0).to(probe_device)
    probe = _build_probe(train_X.shape[1], 1)
    return _train_probe(probe, train_X, train_Y, epochs=300, task="regression")


def train_fused_depth_probe_paths(features, geom_depths, geom_alphas, depth_paths, fH, fW):
    """Train a learned depth-fusion probe and return the full fusion bundle."""
    depth_probe = train_depth_probe_paths(features, depth_paths, fH, fW)
    train_input, train_feat_depth, train_geom_depth, train_geom_valid, train_Y = [], [], [], [], []
    fusion_train_pixel_budget = 100_000
    per_frame_budget = max(256, fusion_train_pixel_budget // max(1, len(depth_paths)))
    fusion_sample_gen = torch.Generator(device="cpu").manual_seed(42)
    if probe_device.type == "cuda":
        torch.cuda.empty_cache()
    for feat, geom, alpha, dpath in zip(features, geom_depths, geom_alphas, depth_paths):
        dpath = Path(dpath)
        if geom is None or not dpath.exists():
            continue
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        d = torch.from_numpy(d.astype(np.float32) / 1000.0)
        d = F.interpolate(d[None, None], (fH, fW), mode="bilinear", align_corners=False).squeeze()
        sample = prepare_depth_fusion_sample(
            feat, geom, alpha, depth_probe, fH, fW, probe_device, output_device="cpu"
        )
        train_sample = sample_depth_fusion_training_pixels(
            sample,
            d,
            max_samples=per_frame_budget,
            generator=fusion_sample_gen,
        )
        if train_sample is None:
            continue
        train_input.append(train_sample["input_flat"])
        train_feat_depth.append(train_sample["feat_depth_flat"])
        train_geom_depth.append(train_sample["geom_depth_flat"])
        train_geom_valid.append(train_sample["geom_valid_flat"])
        train_Y.append(train_sample["targets"])

    if probe_device.type == "cuda":
        torch.cuda.empty_cache()
    if not train_input:
        raise RuntimeError("No valid samples collected for fused depth probe training.")
    fusion_probe = train_depth_fusion_probe(
        torch.cat(train_input, 0),
        torch.cat(train_feat_depth, 0),
        torch.cat(train_geom_depth, 0),
        torch.cat(train_geom_valid, 0),
        torch.cat(train_Y, 0),
        probe_device,
        epochs=300,
    )
    return {"depth_probe": depth_probe, "fusion_probe": fusion_probe}


def train_seg_probe_paths(features, sem_paths, fH, fW):
    """Train a segmentation probe using explicit file paths."""
    train_X, train_Y = [], []
    for feat, spath in zip(features, sem_paths):
        spath = Path(spath)
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

    train_X = torch.cat(train_X, 0).to(probe_device)
    train_Y = torch.cat(train_Y, 0).to(probe_device)
    unique_classes = torch.unique(train_Y).tolist()
    id_to_contiguous = {c: i for i, c in enumerate(unique_classes)}
    contiguous_to_id = {i: c for c, i in id_to_contiguous.items()}
    train_Y = torch.tensor([id_to_contiguous[y.item()] for y in train_Y],
                           dtype=torch.long, device=probe_device)
    n_classes = len(unique_classes)
    counts = torch.bincount(train_Y, minlength=n_classes).float().clamp(min=1)
    weights = (1.0 / counts)
    weights = (weights / weights.sum() * n_classes).to(probe_device)
    probe = _build_probe(train_X.shape[1], n_classes)
    probe = _train_probe(probe, train_X, train_Y, epochs=500, task="classification",
                         class_weights=weights)
    return probe, id_to_contiguous, contiguous_to_id


def predict_fused_depth(probe, feat, geom_depth, geom_alpha, fH, fW):
    """Predict depth using the learned aligned-geometry fusion bundle."""
    sample = prepare_depth_fusion_sample(
        feat,
        geom_depth,
        geom_alpha,
        probe["depth_probe"],
        fH,
        fW,
        probe_device,
    )
    pred = predict_depth_fusion(probe["fusion_probe"], sample, fH, fW)["depth"]
    return pred.cpu().numpy()


# ── SigLIP2 grounding ────────────────────────────────────────────────────────

def load_siglip2_projection(projection_weights, target_device=None):
    """Load SigLIP2 feature projection model.

    Handles both standalone projection weights and full RADIO checkpoint files.
    """
    target_device = target_device or device
    target_dtype = inference_dtype(target_device)
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
    ckpt = torch.load(projection_weights, map_location="cpu")
    # Handle full RADIO checkpoint vs standalone projection weights
    if "state_dict" in ckpt:
        # Full RADIO checkpoint — extract SigLIP2 adaptor head weights
        sd = ckpt["state_dict"]
        prefix = "model.summary.adaptors.siglip2."
        proj_sd = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                proj_sd[new_key] = v
        if not proj_sd:
            raise RuntimeError(f"No SigLIP2 adaptor keys found in checkpoint with prefix '{prefix}'")
        proj.load_state_dict(proj_sd)
    else:
        proj.load_state_dict(ckpt)
    return proj.to(target_device, dtype=target_dtype).eval()


def load_text_embedding_candidates(text_emb_path):
    """Load one or more SigLIP2 text-embedding files as candidate banks."""
    text_emb_path = Path(text_emb_path)
    if text_emb_path.name.startswith("siglip2_text_embeddings"):
        candidate_paths = sorted(
            text_emb_path.parent.glob("siglip2_text_embeddings*.pt"),
            key=lambda p: (
                0 if p.name == "siglip2_text_embeddings_v2.pt" else 1,
                0 if p.name == text_emb_path.name else 1,
                p.name,
            ),
        )
        if not candidate_paths:
            candidate_paths = [text_emb_path]
    else:
        candidate_paths = [text_emb_path]

    candidates = []
    for path in candidate_paths:
        if not path.exists():
            continue
        data = torch.load(str(path), map_location="cpu")
        bank = {
            q: F.normalize(e.float(), dim=0)
            for q, e in zip(data["queries"], data["embeddings"])
        }
        candidates.append((path.name, bank))
    return candidates


def select_scene_text_embeddings(
    candidates,
    proj_model,
    train_gt_feats,
    train_sem_paths,
    active_queries,
    active_query_cids,
    fH,
    fW,
    max_frames=32,
    target_device=None,
):
    """Pick the best embedding bank per query using training GT features/semantics."""
    target_device = target_device or device
    target_dtype = inference_dtype(target_device)
    if not candidates:
        raise ValueError("No text embedding candidates were loaded")
    if len(candidates) == 1:
        name, bank = candidates[0]
        return (
            torch.stack([bank[q] for q in active_queries]).to(target_device, dtype=torch.float32),
            {q: name for q in active_queries},
        )

    projected_feats = []
    projected_sems = []
    for feat, spath in zip(train_gt_feats[:max_frames], train_sem_paths[:max_frames]):
        sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        feat_b = feat.unsqueeze(0).to(target_device, dtype=target_dtype)
        B, C, H, W = feat_b.shape
        sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
        feat_flat = feat_b.reshape(B, C, H * W).permute(0, 2, 1)
        with torch.no_grad():
            siglip = proj_model(feat_flat)
        siglip = F.normalize(siglip.float().squeeze(0), dim=-1)  # [HW, 1536]
        projected_feats.append(siglip)
        projected_sems.append(sem.reshape(-1))

    selected_embeddings = []
    selected_sources = {}
    for query, cid in zip(active_queries, active_query_cids):
        best_name = None
        best_emb = None
        best_score = -float("inf")
        for name, bank in candidates:
            if query not in bank:
                continue
            emb = bank[query].to(target_device, dtype=torch.float32)
            margins = []
            for siglip, sem_flat in zip(projected_feats, projected_sems):
                pos_mask = sem_flat == cid
                if pos_mask.sum() < 10:
                    continue
                pos_mask_t = torch.from_numpy(pos_mask).to(target_device)
                sim = siglip @ emb
                margins.append((sim[pos_mask_t].mean() - sim[~pos_mask_t].mean()).item())
            score = float(np.mean(margins)) if margins else -float("inf")
            if score > best_score:
                best_score = score
                best_name = name
                best_emb = emb
        if best_emb is None:
            best_name, bank = candidates[0]
            best_emb = bank[query].to(target_device, dtype=torch.float32)
        selected_embeddings.append(best_emb)
        selected_sources[query] = f"{best_name} ({best_score:.4f})"

    return torch.stack(selected_embeddings).to(target_device, dtype=torch.float32), selected_sources


def compute_grounding_heatmaps(features_1280, proj_model, text_emb, temperature=1.0, target_device=None):
    """Compute text grounding heatmaps with cosine-softmax normalization.

    Args:
        features_1280: [1, 1280, H, W]
        proj_model: SigLIP2 projection
        text_emb: [K, 1536] normalized text embeddings
        temperature: softmax temperature (1.0 recommended; 0.07 is too aggressive)

    Returns:
        raw_sim: [K, H, W] raw cosine similarity heatmaps
        probs:   [K, H, W] softmax-normalized probabilities across queries
    """
    target_device = target_device or device
    target_dtype = inference_dtype(target_device)
    features_1280 = features_1280.to(target_device, dtype=target_dtype)
    text_emb = text_emb.to(target_device, dtype=target_dtype)
    B, C, H, W = features_1280.shape
    feat_flat = features_1280.reshape(B, C, H * W).permute(0, 2, 1)
    with torch.no_grad():
        siglip = proj_model(feat_flat)
    siglip = F.normalize(siglip.float(), dim=-1).squeeze(0)  # [HW, 1536]

    # Raw cosine similarity (ensure float32 for both operands)
    raw_sim = text_emb.float() @ siglip.float().T  # [K, HW]
    raw_sim = raw_sim.float().reshape(-1, H, W)

    # Softmax across queries with moderate temperature
    K = raw_sim.shape[0]
    sim_flat = raw_sim.reshape(K, -1)  # [K, HW]
    probs = F.softmax(sim_flat / temperature, dim=0)   # softmax across queries
    probs = probs.reshape(raw_sim.shape)               # [K, H, W]

    return raw_sim, probs


# ── Geometry depth helpers ────────────────────────────────────────────────────

def extract_geom_depth_np(geom_depth, alpha_map, fH=None, fW=None):
    """Convert raw geometry depth tensor to numpy [fH, fW] with alpha masking.

    Uses bilateral filter (edge-preserving) to suppress ED alpha-blending
    artifacts while keeping sharp depth discontinuities.
    """
    if geom_depth is None:
        return None
    d = geom_depth
    if isinstance(d, torch.Tensor):
        d = d.detach().cpu()
        if d.dim() == 4:          # [1, 1, H, W]
            d = d.squeeze(0).squeeze(0)
        elif d.dim() == 3:        # [1, H, W]
            d = d.squeeze(0)
        d = d.float().numpy()
    if fH is not None and fW is not None and d.shape != (fH, fW):
        d = cv2.resize(d, (fW, fH), interpolation=cv2.INTER_LINEAR)
    # Hard alpha threshold: mask out low-opacity regions
    valid = np.ones(d.shape, dtype=bool)
    if alpha_map is not None:
        a = alpha_map
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu()
            if a.dim() == 4:
                a = a.squeeze(0).squeeze(0)
            elif a.dim() == 3:
                a = a.squeeze(0)
            a = a.float().numpy()
        if fH is not None and fW is not None and a.shape != (fH, fW):
            a = cv2.resize(a, (fW, fH), interpolation=cv2.INTER_LINEAR)
        valid = a > 0.1
        d[~valid] = 0.0
    # Median filter to remove per-Gaussian salt-and-pepper noise,
    # then light Gaussian blur for smooth appearance.
    d_f32 = d.astype(np.float32)
    if d_f32.max() > 0:
        d_filtered = cv2.medianBlur(d_f32, 3)
        d_filtered = cv2.GaussianBlur(d_filtered, (3, 3), 0.8)
    else:
        d_filtered = d_f32
    d_filtered[~valid] = 0.0
    return d_filtered


def extract_alpha_np(alpha_map, fH=None, fW=None):
    """Convert a geometry alpha map to numpy [fH, fW]."""
    if alpha_map is None:
        return None
    a = alpha_map
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu()
        if a.dim() == 4:
            a = a.squeeze(0).squeeze(0)
        elif a.dim() == 3:
            a = a.squeeze(0)
        a = a.float().numpy()
    if fH is not None and fW is not None and a.shape != (fH, fW):
        a = cv2.resize(a, (fW, fH), interpolation=cv2.INTER_LINEAR)
    return np.clip(a.astype(np.float32), 0.0, 1.0)


def align_depth_scale_shift(pred, gt, valid_mask=None):
    """Least-squares scale-shift alignment: gt ≈ scale * pred + shift."""
    if valid_mask is None:
        valid_mask = gt > 0.01
    if valid_mask.sum() < 10:
        return pred
    p_vals = pred[valid_mask].astype(np.float64)
    g_vals = gt[valid_mask].astype(np.float64)
    A = np.stack([p_vals, np.ones_like(p_vals)], axis=1)
    result = np.linalg.lstsq(A, g_vals, rcond=None)
    scale, shift = result[0]
    aligned = pred.astype(np.float64) * scale + shift
    return aligned.astype(np.float32)


# ── Main visualization ────────────────────────────────────────────────────────

def main():
    global probe_device
    parser = argparse.ArgumentParser(description="RADIO-GS Visualization v2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="/root/results/v2")
    parser.add_argument("--num_views", type=int, default=10,
                        help="Number of novel-view frames to visualize")
    parser.add_argument("--n_train", type=int, default=300,
                        help="Number of training frames for probes")
    parser.add_argument("--scale", type=int, default=16,
                        help="Upscale factor for feature-resolution images")
    parser.add_argument("--text_embeddings",
                        default=DEFAULT_SIGLIP2_TEXT_EMBEDDINGS)
    parser.add_argument("--projection_weights",
                        default="output/radio_gs/siglip2_feat_projection.pth")
    parser.add_argument("--grounding_queries", nargs="+",
                        default=list(GROUNDING_QUERY_CLASS_IDS.keys()))
    parser.add_argument("--grounding_seg_threshold", type=float, default=0.35,
                        help="Confidence threshold for text-derived grounding masks")
    parser.add_argument("--grounding_device", choices=["cpu", "cuda"], default="cpu",
                        help="Device for qualitative grounding visualization")
    parser.add_argument("--probe_device", choices=["cpu", "cuda"], default="cpu",
                        help="Device for visualization probe training/prediction")
    args = parser.parse_args()

    S = args.scale
    grounding_device = torch.device(args.grounding_device)
    probe_device = torch.device(args.probe_device)
    out_root = Path(args.output_dir)

    # Create output subdirectories
    dirs = {}
    for name in ["feature_pca", "depth", "segmentation",
                  "grounding", "grounding_seg", "composite"]:
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d

    # Load pipeline
    print("Loading RADIO-GS pipeline...")
    model, codec, renderer, sharpener, refiner, config, is_hybrid = load_pipeline(
        args.config, args.checkpoint)

    scene = getattr(config, "scene", "room_0")
    scene_root = Path("dataset") / scene
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")
    train_feat_dir = resolve_split_feature_dir(config, "train")
    val_feat_dir = resolve_split_feature_dir(config, "val")
    train_feat_root = train_feat_dir / "backbone" if (train_feat_dir / "backbone").exists() else train_feat_dir
    val_feat_root = val_feat_dir / "backbone" if (val_feat_dir / "backbone").exists() else val_feat_dir
    fH = getattr(config, "feature_height", 30)
    fW = getattr(config, "feature_width", 40)
    imgH = getattr(config, "image_height", 480)
    imgW = getattr(config, "image_width", 640)
    geom_renderer = build_renderer(config, image_height=imgH, image_width=imgW)

    # ── Handle mixed_split (rooms 1&2 use random 80/20 split across sequences) ──
    mixed_split = getattr(config, "mixed_split", False)
    if mixed_split:
        mixed_ratio = getattr(config, "mixed_train_ratio", 0.8)
        mixed_seed = getattr(config, "mixed_seed", 42)
        # Load poses from both sequences
        poses_s1 = np.loadtxt(str(scene_root / train_split / "traj_w_c.txt")).reshape(-1, 4, 4).astype(np.float32)
        poses_s2 = np.loadtxt(str(scene_root / val_split / "traj_w_c.txt")).reshape(-1, 4, 4).astype(np.float32)
        n_s1 = len(poses_s1)
        n_s2 = len(poses_s2)
        total = n_s1 + n_s2
        train_size = int(mixed_ratio * total)
        gen = torch.Generator().manual_seed(mixed_seed)
        perm = torch.randperm(total, generator=gen).tolist()
        train_mixed = sorted(perm[:train_size])
        val_mixed = sorted(perm[train_size:])

        def _split_idx(combined_idx):
            """Map combined index → (sequence_name, frame_idx_in_sequence)."""
            if combined_idx < n_s1:
                return train_split, combined_idx
            return val_split, combined_idx - n_s1

        w2c_s1 = np.linalg.inv(poses_s1)
        w2c_s2 = np.linalg.inv(poses_s2)

        def _get_w2c(seq, fidx):
            return w2c_s1[fidx] if seq == train_split else w2c_s2[fidx]

        # Select vis frames from the val portion of the mixed split
        vis_step = max(1, len(val_mixed) // args.num_views)
        vis_mixed = val_mixed[::vis_step][:args.num_views]
        vis_seq_frame = [_split_idx(ci) for ci in vis_mixed]

        # For training probes, sample from train portion
        train_step = max(1, len(train_mixed) // args.n_train)
        train_mixed_sub = train_mixed[::train_step][:args.n_train]
        train_seq_frame = [_split_idx(ci) for ci in train_mixed_sub]

        print(f"Mixed split: total={total}, train={len(train_mixed)}, val={len(val_mixed)}")
        print(f"Visualizing {len(vis_seq_frame)} val frames, training probes on {len(train_seq_frame)} frames")
    else:
        # Standard sequential split
        val_pose_file = scene_root / val_split / "traj_w_c.txt"
        all_c2w = np.loadtxt(str(val_pose_file)).reshape(-1, 4, 4).astype(np.float32)
        all_w2c = np.linalg.inv(all_c2w)
        n_total = len(all_w2c)
        vis_indices = list(range(0, n_total, max(1, n_total // args.num_views)))[:args.num_views]
        print(f"Visualizing {len(vis_indices)} novel views from {val_split}")

    # Helper functions for loading GT data (unified for both split modes)
    def get_vis_w2c(j):
        """Get w2c matrix for visualization frame j."""
        if mixed_split:
            seq, fidx = vis_seq_frame[j]
            return _get_w2c(seq, fidx)
        return all_w2c[vis_indices[j]]

    def get_vis_gt_feat_path(j):
        """Get GT feature path for visualization frame j."""
        if mixed_split:
            seq, fidx = vis_seq_frame[j]
            feat_root = train_feat_root if seq == train_split else val_feat_root
            return feat_root / f"rgb_{fidx}.pt"
        return val_feat_root / f"rgb_{vis_indices[j]}.pt"

    def get_vis_rgb_path(j):
        """Get RGB image path for visualization frame j."""
        if mixed_split:
            seq, fidx = vis_seq_frame[j]
            return scene_root / seq / "rgb" / f"rgb_{fidx}.png"
        return scene_root / val_split / "rgb" / f"rgb_{vis_indices[j]}.png"

    def get_vis_depth_path(j):
        """Get depth map path for visualization frame j."""
        if mixed_split:
            seq, fidx = vis_seq_frame[j]
            return scene_root / seq / "depth" / f"depth_{fidx}.png"
        return scene_root / val_split / "depth" / f"depth_{vis_indices[j]}.png"

    def get_vis_sem_path(j):
        """Get semantic label path for visualization frame j."""
        if mixed_split:
            seq, fidx = vis_seq_frame[j]
            return scene_root / seq / "semantic_class" / f"semantic_class_{fidx}.png"
        return scene_root / val_split / "semantic_class" / f"semantic_class_{vis_indices[j]}.png"

    def get_vis_label(j):
        """Get a short label for the frame (for filenames)."""
        if mixed_split:
            seq, fidx = vis_seq_frame[j]
            return f"{seq}_f{fidx}"
        return f"f{vis_indices[j]:04d}"

    rgb_cache = {}
    depth_cache = {}
    sem_cache = {}

    def load_vis_rgb(j):
        if j not in rgb_cache:
            rgb = cv2.imread(str(get_vis_rgb_path(j)))
            rgb_cache[j] = None if rgb is None else cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb_cache[j]

    def load_vis_depth(j):
        if j not in depth_cache:
            depth_cache[j] = cv2.imread(str(get_vis_depth_path(j)), cv2.IMREAD_UNCHANGED)
        return depth_cache[j]

    def load_vis_sem(j):
        if j not in sem_cache:
            sem_cache[j] = cv2.imread(str(get_vis_sem_path(j)), cv2.IMREAD_GRAYSCALE)
        return sem_cache[j]

    n_vis = len(vis_seq_frame) if mixed_split else len(vis_indices)

    # ── Step 1: Render all features ──────────────────────────────────────────
    print(f"\n[1/7] Rendering decoded features for {n_vis} visualization frames...")
    gt_feats, rend_feats = [], []
    geom_depths = []        # full-resolution geometry depth from 3DGS per vis frame
    geom_depths_lowres = [] # feature-resolution geometry depth for fusion
    geom_alphas_lowres = []
    with torch.no_grad():
        for j in tqdm(range(n_vis), desc="Rendering"):
            gt_path = get_vis_gt_feat_path(j)
            gt = torch.load(str(gt_path), map_location="cpu").float()
            if gt.dim() == 2:
                # Features stored at native RADIO resolution (e.g. 30x40)
                n_pixels = gt.shape[0]
                C = gt.shape[1]
                native_h = int(round((n_pixels * fH / fW) ** 0.5))
                native_w = n_pixels // native_h
                if native_h * native_w != n_pixels:
                    native_h, native_w = fH, fW
                gt = gt.reshape(native_h, native_w, C).permute(2, 0, 1)
                if native_h != fH or native_w != fW:
                    gt = F.interpolate(gt.unsqueeze(0), (fH, fW),
                                       mode="bilinear", align_corners=False).squeeze(0)
            elif gt.shape[-2] != fH or gt.shape[-1] != fW:
                gt = F.interpolate(gt.unsqueeze(0), (fH, fW),
                                   mode="bilinear", align_corners=False).squeeze(0)
            gt_feats.append(gt)

            w2c = get_vis_w2c(j)
            pose = torch.from_numpy(w2c[np.newaxis]).to(device)
            decoded, geom_depth, alpha_map = render_features(
                model, codec, renderer, sharpener, refiner, config, pose,
                is_hybrid=is_hybrid)
            rend_feats.append(decoded.squeeze(0).cpu())
            geom_depths_lowres.append(
                extract_geom_depth_np(geom_depth, alpha_map, fH, fW))
            geom_alphas_lowres.append(
                extract_alpha_np(alpha_map, fH, fW))
            geom_full = geom_renderer.render_rgb(model, pose.squeeze(0))
            geom_depths.append(
                extract_geom_depth_np(geom_full["depth"], geom_full["alpha"]))

    # ── Step 2: Train probes on training data ────────────────────────────────
    print("\n[2/7] Training linear probes on training split features...")
    if mixed_split:
        n_train_use = len(train_seq_frame)
        train_depth_paths = [scene_root / seq / "depth" / f"depth_{fidx}.png"
                             for seq, fidx in train_seq_frame]
        train_sem_paths = [scene_root / seq / "semantic_class" / f"semantic_class_{fidx}.png"
                           for seq, fidx in train_seq_frame]
    else:
        train_pose_file = scene_root / train_split / "traj_w_c.txt"
        train_c2w = np.loadtxt(str(train_pose_file)).reshape(-1, 4, 4).astype(np.float32)
        train_w2c_all = np.linalg.inv(train_c2w)
        n_train_total = len(train_w2c_all)
        train_indices = list(range(0, n_train_total,
                                   max(1, n_train_total // args.n_train)))[:args.n_train]
        n_train_use = len(train_indices)
        train_depth_paths = [scene_root / train_split / "depth" / f"depth_{i}.png"
                             for i in train_indices]
        train_sem_paths = [scene_root / train_split / "semantic_class" / f"semantic_class_{i}.png"
                           for i in train_indices]

    # Render training features
    train_gt_feats, train_rend_feats = [], []
    train_geom_depths_lowres = []
    train_geom_alphas_lowres = []

    print(f"  Rendering {n_train_use} training features...")
    with torch.no_grad():
        for j in tqdm(range(n_train_use), desc="Train render", leave=False):
            if mixed_split:
                seq, fidx = train_seq_frame[j]
                feat_root = train_feat_root if seq == train_split else val_feat_root
                gt_path = feat_root / f"rgb_{fidx}.pt"
                w2c = _get_w2c(seq, fidx)
            else:
                gt_path = train_feat_root / f"rgb_{train_indices[j]}.pt"
                w2c = train_w2c_all[train_indices[j]]
            gt = torch.load(str(gt_path), map_location="cpu").float()
            if gt.dim() == 2:
                n_pixels = gt.shape[0]
                C = gt.shape[1]
                native_h = int(round((n_pixels * fH / fW) ** 0.5))
                native_w = n_pixels // native_h
                if native_h * native_w != n_pixels:
                    native_h, native_w = fH, fW
                gt = gt.reshape(native_h, native_w, C).permute(2, 0, 1)
                if native_h != fH or native_w != fW:
                    gt = F.interpolate(gt.unsqueeze(0), (fH, fW),
                                       mode="bilinear", align_corners=False).squeeze(0)
            elif gt.shape[-2] != fH or gt.shape[-1] != fW:
                gt = F.interpolate(gt.unsqueeze(0), (fH, fW),
                                   mode="bilinear", align_corners=False).squeeze(0)
            train_gt_feats.append(gt)

            pose = torch.from_numpy(w2c[np.newaxis]).to(device)
            decoded, geom_depth, alpha_map = render_features(
                model, codec, renderer, sharpener, refiner, config, pose,
                is_hybrid=is_hybrid)
            train_rend_feats.append(decoded.squeeze(0).cpu())
            train_geom_depths_lowres.append(
                extract_geom_depth_np(geom_depth, alpha_map, fH, fW))
            train_geom_alphas_lowres.append(
                extract_alpha_np(alpha_map, fH, fW))

    # Train probes using path-based loaders (compatible with mixed_split)
    print("  Training oracle depth probe...")
    oracle_depth_probe = train_depth_probe_paths(train_gt_feats, train_depth_paths, fH, fW)
    print("  Training oracle segmentation probe...")
    oracle_seg_probe, oracle_seg_id_to_contig, oracle_seg_contig_to_id = train_seg_probe_paths(
        train_gt_feats, train_sem_paths, fH, fW)
    print("  Training rendered depth probe...")
    rend_depth_probe = train_depth_probe_paths(train_rend_feats, train_depth_paths, fH, fW)
    print("  Training fused depth probe...")
    fused_depth_probe = train_fused_depth_probe_paths(
        train_rend_feats, train_geom_depths_lowres, train_geom_alphas_lowres,
        train_depth_paths, fH, fW)
    print("  Training rendered segmentation probe...")
    rend_seg_probe, rend_seg_id_to_contig, rend_seg_contig_to_id = train_seg_probe_paths(
        train_rend_feats, train_sem_paths, fH, fW)

    # Cache task predictions used by multiple visualization stages
    tH, tW = fH * S, fW * S
    oracle_depth_preds = []
    rend_depth_preds = []
    fused_depth_preds = []
    oracle_seg_preds_hr = []
    rend_seg_preds_hr = []
    for j in range(n_vis):
        oracle_depth_preds.append(predict_depth(oracle_depth_probe, gt_feats[j], fH, fW))
        rend_depth_preds.append(predict_depth(rend_depth_probe, rend_feats[j], fH, fW))
        fused_depth_preds.append(
            predict_fused_depth(
                fused_depth_probe, rend_feats[j], geom_depths_lowres[j],
                geom_alphas_lowres[j], fH, fW)
        )
        oracle_seg_preds_hr.append(
            predict_seg_smooth(
                oracle_seg_probe, gt_feats[j], fH, fW, tH, tW,
                contiguous_to_id=oracle_seg_contig_to_id,
            )
        )
        rend_seg_preds_hr.append(
            predict_seg_smooth(
                rend_seg_probe, rend_feats[j], fH, fW, tH, tW,
                contiguous_to_id=rend_seg_contig_to_id,
            )
        )

    if device.type == "cuda":
        for module in [
            model, codec, renderer, geom_renderer, sharpener, refiner,
            oracle_depth_probe, rend_depth_probe, fused_depth_probe,
            oracle_seg_probe, rend_seg_probe,
        ]:
            if isinstance(module, nn.Module):
                module.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        print("  Released feature-field/probe GPU state before CPU-heavy visualization stages")

    # ── Step 3: Feature PCA visualization ────────────────────────────────────
    print("\n[3/7] Generating feature PCA visualizations...")
    all_for_pca = gt_feats + rend_feats
    pca_imgs = shared_pca_colorize(all_for_pca, n_components=3)
    gt_pcas = pca_imgs[:len(gt_feats)]
    rend_pcas = pca_imgs[len(gt_feats):]

    for j in range(n_vis):
        gt_f, rend_f = gt_feats[j], rend_feats[j]
        # Normalize along channel dim before cosine (matches training loss)
        rend_norm = F.normalize(rend_f, p=2, dim=0)
        gt_norm = F.normalize(gt_f, p=2, dim=0)
        cos = F.cosine_similarity(
            rend_norm.reshape(-1, fH * fW), gt_norm.reshape(-1, fH * fW), dim=0
        ).reshape(fH, fW).numpy()

        rgb = load_vis_rgb(j)
        tH, tW = fH * S, fW * S
        rgb_display = cv2.resize(rgb, (tW, tH), interpolation=cv2.INTER_LINEAR)

        panels = [
            rgb_display,
            upscale(gt_pcas[j], S),
            upscale(rend_pcas[j], S),
            upscale(cosine_map_to_heatmap(cos), S),
        ]

        labels = ["Input RGB", "GT Feature (PCA)", "Rendered Feature (PCA)",
                   f"Cosine Sim (mean={cos.mean():.3f})"]
        for k, label in enumerate(labels):
            panels[k] = add_text(panels[k], label, pos=(5, 20), font_scale=0.55)

        row = hconcat_with_border(panels, border=3)
        save_path = dirs["feature_pca"] / f"pca_frame_{get_vis_label(j)}_cos{cos.mean():.3f}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # PCA summary grid (up to 8 frames)
    n_grid = min(8, n_vis)
    grid_rows = []
    for j in range(n_grid):
        idx = get_vis_label(j)
        rend_norm_g = F.normalize(rend_feats[j], p=2, dim=0)
        gt_norm_g = F.normalize(gt_feats[j], p=2, dim=0)
        cos = F.cosine_similarity(
            rend_norm_g.reshape(-1, fH * fW), gt_norm_g.reshape(-1, fH * fW), dim=0
        ).reshape(fH, fW).numpy()
        rgb = cv2.imread(str(get_vis_rgb_path(j)))
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
    print(f"  Saved PCA grid and {n_vis} individual frames")

    # ── Step 4: Depth visualization (with geometry depth) ────────────────────
    print("\n[4/7] Generating depth estimation visualizations...")
    for j in range(n_vis):
        # GT depth
        dpath = get_vis_depth_path(j)
        d_raw = load_vis_depth(j)
        if d_raw is None:
            continue
        gt_depth = d_raw.astype(np.float32) / 1000.0
        gt_vmin = gt_depth[gt_depth > 0.01].min() if (gt_depth > 0.01).any() else 0
        gt_vmax = gt_depth.max()

        # Geometry depth from 3DGS (full-res + scale/shift aligned to GT)
        gd = geom_depths[j]
        has_geom = gd is not None and gd.max() > 0.01
        if has_geom:
            gd_full = align_depth_scale_shift(gd, gt_depth)
            gd_full = smooth_depth_for_display(gd_full, gd_full > 0.01)
            gd_panel = cv2.resize(
                depth_to_colormap(gd_full, gt_vmin, gt_vmax), (tW, tH),
                interpolation=cv2.INTER_LINEAR,
            )

        oracle_pred = resize_depth_map(oracle_depth_preds[j], gt_depth.shape[:2])
        rend_pred = resize_depth_map(rend_depth_preds[j], gt_depth.shape[:2])
        fused_pred = resize_depth_map(fused_depth_preds[j], gt_depth.shape[:2])
        rgb = load_vis_rgb(j)
        rgb_display = cv2.resize(rgb, (tW, tH), interpolation=cv2.INTER_LINEAR)

        def _resize_panel(img):
            """Ensure panel matches display resolution tH×tW."""
            if img.shape[0] != tH or img.shape[1] != tW:
                return cv2.resize(img, (tW, tH), interpolation=cv2.INTER_LINEAR)
            return img

        # 7 panels: RGB | GT Depth(full-res) | Geom Depth | Oracle Pred | Feature Pred | Fused Pred | Error Map
        panels = [
            rgb_display,
            _resize_panel(depth_to_colormap(gt_depth, gt_vmin, gt_vmax)),
        ]

        if has_geom:
            panels.append(_resize_panel(gd_panel))
        else:
            panels.append(np.zeros((tH, tW, 3), dtype=np.uint8))

        panels += [
            _resize_panel(depth_to_colormap(oracle_pred, gt_vmin, gt_vmax)),
            _resize_panel(depth_to_colormap(rend_pred, gt_vmin, gt_vmax)),
            _resize_panel(depth_to_colormap(fused_pred, gt_vmin, gt_vmax)),
            _resize_panel(depth_error_map(fused_pred, gt_depth)),
        ]

        labels = ["RGB", "GT Depth (Full)", "Geom Depth", "Oracle Pred",
                   "Feature Pred", "Fused Pred", "Fused Error"]
        for k, label in enumerate(labels):
            panels[k] = add_text(panels[k], label, pos=(5, 20), font_scale=0.5)

        row = hconcat_with_border(panels, border=3)
        save_path = dirs["depth"] / f"depth_frame_{get_vis_label(j)}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # Depth grid
    grid_rows = []
    for j in range(min(8, n_vis)):
        idx = get_vis_label(j)
        dpath = get_vis_depth_path(j)
        d_raw = load_vis_depth(j)
        if d_raw is None:
            continue
        gt_d = d_raw.astype(np.float32) / 1000.0
        gt_vmin = gt_d[gt_d > 0.01].min() if (gt_d > 0.01).any() else 0
        gt_vmax = gt_d.max()

        gd = geom_depths[j]
        has_geom = gd is not None and gd.max() > 0.01
        if has_geom:
            gd = align_depth_scale_shift(gd, gt_d)
            gd = smooth_depth_for_display(gd, gd > 0.01)

        r_pred = resize_depth_map(rend_depth_preds[j], gt_d.shape[:2])
        f_pred = resize_depth_map(fused_depth_preds[j], gt_d.shape[:2])
        rgb = load_vis_rgb(j)

        geom_panel = (cv2.resize(depth_to_colormap(gd, gt_vmin, gt_vmax), (tW, tH),
                                 interpolation=cv2.INTER_LINEAR)
                      if has_geom
                      else np.zeros((tH, tW, 3), dtype=np.uint8))

        def _dgrid(img):
            if img.shape[0] != tH or img.shape[1] != tW:
                return cv2.resize(img, (tW, tH), interpolation=cv2.INTER_LINEAR)
            return img

        panels = [
            cv2.resize(rgb, (tW, tH), interpolation=cv2.INTER_LINEAR),
            _dgrid(depth_to_colormap(gt_d, gt_vmin, gt_vmax)),
            geom_panel,
            _dgrid(depth_to_colormap(r_pred, gt_vmin, gt_vmax)),
            _dgrid(depth_to_colormap(f_pred, gt_vmin, gt_vmax)),
            _dgrid(depth_error_map(f_pred, gt_d)),
        ]
        grid_rows.append(hconcat_with_border(panels, border=2))

    if grid_rows:
        header = make_header(
            ["Input RGB", "GT Depth (Full)", "Geom Depth", "Feature Pred", "Fused Pred", "Fused Error"],
            fW * S, height=30, border=2)
        full_grid = vconcat_with_border([header] + grid_rows, border=2)
        cv2.imwrite(str(dirs["depth"] / "depth_grid.png"),
                    cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved depth grid and {n_vis} individual frames")

    # ── Step 5: Segmentation visualization ───────────────────────────────────
    print("\n[5/7] Generating segmentation visualizations...")
    for j in range(n_vis):
        sem_raw = load_vis_sem(j)
        if sem_raw is None:
            continue
        # Work at display resolution for smooth boundaries
        gt_sem_hr = cv2.resize(sem_raw, (tW, tH), interpolation=cv2.INTER_NEAREST)

        oracle_seg_hr = oracle_seg_preds_hr[j]
        rend_seg_hr = rend_seg_preds_hr[j]

        rgb = load_vis_rgb(j)
        rgb_hr = cv2.resize(rgb, (tW, tH))

        alpha = 0.6
        gt_blend = (alpha * seg_to_color(gt_sem_hr) + (1 - alpha) * rgb_hr).astype(np.uint8)
        oracle_blend = (alpha * seg_to_color(oracle_seg_hr) + (1 - alpha) * rgb_hr).astype(np.uint8)
        rend_blend = (alpha * seg_to_color(rend_seg_hr) + (1 - alpha) * rgb_hr).astype(np.uint8)

        panels = [
            cv2.resize(rgb, (tW, tH)),
            gt_blend,
            oracle_blend,
            rend_blend,
        ]

        labels = ["Input RGB", "GT Segmentation", "Oracle Pred", "Rendered Pred"]
        for k, label in enumerate(labels):
            panels[k] = add_text(panels[k], label, pos=(5, 20), font_scale=0.5)

        row = hconcat_with_border(panels, border=3)
        save_path = dirs["segmentation"] / f"seg_frame_{get_vis_label(j)}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # Segmentation grid
    grid_rows = []
    for j in range(min(8, n_vis)):
        idx = get_vis_label(j)
        sem_raw = load_vis_sem(j)
        if sem_raw is None:
            continue
        gt_sem_hr = cv2.resize(sem_raw, (tW, tH), interpolation=cv2.INTER_NEAREST)
        r_seg_hr = rend_seg_preds_hr[j]
        rgb = load_vis_rgb(j)
        rgb_hr = cv2.resize(rgb, (tW, tH))
        gt_blend = (0.6 * seg_to_color(gt_sem_hr) + 0.4 * rgb_hr).astype(np.uint8)
        rend_blend = (0.6 * seg_to_color(r_seg_hr) + 0.4 * rgb_hr).astype(np.uint8)
        panels = [rgb_hr, gt_blend, rend_blend]
        grid_rows.append(hconcat_with_border(panels, border=2))

    if grid_rows:
        header = make_header(["Input RGB", "GT Segmentation", "Rendered Pred"],
                             fW * S, height=30, border=2)
        full_grid = vconcat_with_border([header] + grid_rows, border=2)
        cv2.imwrite(str(dirs["segmentation"] / "seg_grid.png"),
                    cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved segmentation grid and {n_vis} individual frames")

    # ── Step 6: Text grounding heatmaps (per-query normalized + softmax) ─────
    print("\n[6/7] Generating text grounding heatmaps...")
    text_emb_path = Path(args.text_embeddings)
    proj_path = Path(args.projection_weights)
    if text_emb_path.exists() and proj_path.exists():
        candidate_banks = load_text_embedding_candidates(text_emb_path)
        if not candidate_banks:
            print("  ⚠ Skipping grounding: no text embedding files could be loaded")
            candidate_banks = []
        all_queries = sorted({q for _, bank in candidate_banks for q in bank.keys()})
        query_to_idx = {q: i for i, q in enumerate(all_queries)}

        proj_model = load_siglip2_projection(str(proj_path), grounding_device)

        scene_present_ids = set()
        for spath in train_sem_paths[: min(len(train_sem_paths), 64)]:
            sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
            if sem is not None:
                scene_present_ids.update(np.unique(sem).tolist())
        for j in range(n_vis):
            sem = load_vis_sem(j)
            if sem is not None:
                scene_present_ids.update(np.unique(sem).tolist())

        # Filter to requested queries that are both embedded and present in scene semantics
        active_queries = [
            q for q in args.grounding_queries
            if q in query_to_idx and GROUNDING_QUERY_CLASS_IDS.get(q) in scene_present_ids
        ]
        # Sort by class ID for deterministic ordering (matches eval_grounding.py)
        active_queries.sort(key=lambda q: GROUNDING_QUERY_CLASS_IDS[q])
        dropped_queries = [q for q in args.grounding_queries if q not in active_queries]
        if dropped_queries:
            print(f"  Skipping absent/unmapped queries: {dropped_queries}")
        if not active_queries:
            print("  ⚠ No requested grounding queries are present in this scene")
        else:
            active_query_cids = [GROUNDING_QUERY_CLASS_IDS[q] for q in active_queries]
            active_text_emb, text_sources = select_scene_text_embeddings(
                candidate_banks,
                proj_model,
                train_gt_feats,
                train_sem_paths,
                active_queries,
                active_query_cids,
                fH,
                fW,
                target_device=grounding_device,
            )
            print("  Selected text embeddings:")
            for q in active_queries:
                print(f"    {q}: {text_sources[q]}")

        # Assign a color per active query for softmax segmentation
        query_colors = {}
        np.random.seed(123)
        for qi, qname in enumerate(active_queries):
            if qname in GROUNDING_QUERY_CLASS_IDS and GROUNDING_QUERY_CLASS_IDS[qname] in SEG_COLORS:
                query_colors[qi] = SEG_COLORS[GROUNDING_QUERY_CLASS_IDS[qname]]
            else:
                query_colors[qi] = tuple(np.random.randint(60, 255, 3).tolist())

        print(f"  Active grounding queries: {active_queries}")

        for j in range(n_vis):
            if not active_queries:
                break
            gt_f = gt_feats[j].unsqueeze(0)
            rend_f = rend_feats[j].unsqueeze(0)

            gt_raw, gt_probs = compute_grounding_heatmaps(
                gt_f, proj_model, active_text_emb, target_device=grounding_device)
            rend_raw, rend_probs = compute_grounding_heatmaps(
                rend_f, proj_model, active_text_emb, target_device=grounding_device)
            rend_probs_np = rend_probs.cpu().numpy()

            # Load semantic GT for mask overlay
            sem_raw = load_vis_sem(j)
            if sem_raw is not None:
                gt_sem = cv2.resize(sem_raw, (fW, fH), interpolation=cv2.INTER_NEAREST)
                gt_sem_hr = cv2.resize(sem_raw, (tW, tH), interpolation=cv2.INTER_NEAREST)
            else:
                gt_sem = None
                gt_sem_hr = None

            rgb = load_vis_rgb(j)
            rgb_hr = cv2.resize(rgb, (tW, tH), interpolation=cv2.INTER_LINEAR)

            # Per-query rows: GT Semantic Mask | Rendered Text Mask | Rendered Heatmap
            if j == 0:
                print(f"  [DEBUG] Frame {get_vis_label(j)} grounding correspondence:")
                for qi_dbg, q_dbg in enumerate(active_queries):
                    cid_dbg = active_query_cids[qi_dbg]
                    gt_area = int((gt_sem == cid_dbg).sum()) if gt_sem is not None else 0
                    print(f"    qi={qi_dbg}: query='{q_dbg}' class_id={cid_dbg} gt_pixels={gt_area}")
            query_rows = []
            for qi, qname in enumerate(active_queries):
                gt_h = gt_raw[qi].cpu().numpy()
                rend_h = rend_raw[qi].cpu().numpy()

                # Per-query percentile normalization for better contrast
                def _normalize(arr, plo=2, phi=98):
                    lo = np.percentile(arr, plo)
                    hi = np.percentile(arr, phi)
                    if hi - lo > 1e-8:
                        return np.clip((arr - lo) / (hi - lo), 0, 1)
                    return np.zeros_like(arr)

                gt_norm = _normalize(gt_h)
                rend_norm = _normalize(rend_h)

                # Upscale heatmaps to display resolution (bilinear for smooth gradients)
                gt_norm_hr = cv2.resize(gt_norm, (tW, tH), interpolation=cv2.INTER_LINEAR)
                rend_norm_hr = cv2.resize(rend_norm, (tW, tH), interpolation=cv2.INTER_LINEAR)

                gt_color = cv2.applyColorMap(
                    (gt_norm_hr * 255).astype(np.uint8), cv2.COLORMAP_JET)
                gt_color = cv2.cvtColor(gt_color, cv2.COLOR_BGR2RGB)
                rend_color = cv2.applyColorMap(
                    (rend_norm_hr * 255).astype(np.uint8), cv2.COLORMAP_JET)
                rend_color = cv2.cvtColor(rend_color, cv2.COLOR_BGR2RGB)

                # GT mask from semantic labels, not from oracle heatmaps.
                gt_mask_thresh = (
                    (gt_sem_hr == active_query_cids[qi]).astype(np.uint8)
                    if gt_sem_hr is not None else np.zeros((tH, tW), dtype=np.uint8)
                )
                mask_vis = np.zeros((tH, tW, 3), dtype=np.uint8)
                mask_vis[gt_mask_thresh > 0] = (0, 255, 100)
                mask_blend = (0.5 * mask_vis + 0.5 * rgb_hr).astype(np.uint8)

                text_mask_lr = grounding_mask_from_probs(
                    rend_probs_np[qi], rend_probs_np, qi,
                    base_threshold=args.grounding_seg_threshold,
                )
                pred_mask = cv2.resize(text_mask_lr, (tW, tH), interpolation=cv2.INTER_NEAREST)
                pred_vis = np.zeros((tH, tW, 3), dtype=np.uint8)
                pred_vis[pred_mask > 0] = query_colors[qi]
                pred_blend = (0.5 * pred_vis + 0.5 * rgb_hr).astype(np.uint8)

                rend_overlay = (0.5 * rend_color + 0.5 * rgb_hr).astype(np.uint8)

                panels = [mask_blend, pred_blend, rend_overlay]
                panels[0] = add_text(panels[0], qname, pos=(5, 20), font_scale=0.55)
                query_rows.append(hconcat_with_border(panels, border=2))

            if query_rows:
                header = make_header(
                    ["GT Semantic Mask", "Rendered Text Mask", "Rendered Heatmap"],
                    tW, height=28, border=2)
                rgb_labeled = add_text(rgb_hr.copy(), f"Frame {get_vis_label(j)}", pos=(5, 20), font_scale=0.55)
                rgb_row = np.zeros((rgb_hr.shape[0], header.shape[1], 3), dtype=np.uint8)
                x_off = (rgb_row.shape[1] - rgb_hr.shape[1]) // 2
                rgb_row[:, x_off:x_off + rgb_hr.shape[1]] = rgb_labeled

                full = vconcat_with_border([rgb_row, header] + query_rows, border=2)
                save_path = dirs["grounding"] / f"grounding_frame_{get_vis_label(j)}.png"
                cv2.imwrite(str(save_path), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))

            # ── Zero-shot segmentation from softmax grounding ────────────
            # Upscale grounding probabilities bilinearly before argmax
            rend_seg_vis = np.zeros((tH, tW, 3), dtype=np.uint8)
            for qi, cid in enumerate(active_query_cids):
                rend_seg_vis[rend_seg_preds_hr[j] == cid] = query_colors[qi]

            # GT segmentation restricted to the chosen active semantic classes
            gt_seg_vis = np.zeros((tH, tW, 3), dtype=np.uint8)
            if gt_sem_hr is not None:
                for qi, cid in enumerate(active_query_cids):
                    gt_seg_vis[gt_sem_hr == cid] = query_colors[qi]

            panels = [rgb_hr, gt_seg_vis, rend_seg_vis]
            seg_labels = ["RGB", "GT Query Seg", "Rendered Query Seg"]
            for k, label in enumerate(seg_labels):
                panels[k] = add_text(panels[k], label, pos=(5, 20), font_scale=0.5)
            seg_row = hconcat_with_border(panels, border=3)
            save_path = dirs["grounding_seg"] / f"grounding_seg_frame_{get_vis_label(j)}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(seg_row, cv2.COLOR_RGB2BGR))

        # Grounding segmentation grid
        seg_grid_rows = []
        for j in range(min(8, n_vis)):
            p = dirs["grounding_seg"] / f"grounding_seg_frame_{get_vis_label(j)}.png"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    seg_grid_rows.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if seg_grid_rows:
            full_grid = vconcat_with_border(seg_grid_rows, border=2)
            cv2.imwrite(str(dirs["grounding_seg"] / "grounding_seg_grid.png"),
                        cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR))

        if active_queries:
            print(f"  Saved grounding + grounding_seg for {n_vis} frames")
        else:
            print("  ⚠ Skipping grounding panels: no active queries after scene filtering")
    else:
        print("  ⚠ Skipping grounding: text embeddings or projection weights not found")

    # ── Step 7: Composite multi-task figure ──────────────────────────────────
    print("\n[7/7] Generating composite multi-task figures...")
    for j in range(min(5, n_vis)):  # Top 5 frames
        rgb = load_vis_rgb(j)
        tH_c, tW_c = fH * S, fW * S
        rgb_display = cv2.resize(rgb, (tW_c, tH_c), interpolation=cv2.INTER_LINEAR)

        # PCA + cosine (normalize along channel dim to match training)
        rend_norm_c = F.normalize(rend_feats[j], p=2, dim=0)
        gt_norm_c = F.normalize(gt_feats[j], p=2, dim=0)
        cos = F.cosine_similarity(
            rend_norm_c.reshape(-1, fH * fW), gt_norm_c.reshape(-1, fH * fW), dim=0
        ).reshape(fH, fW).numpy()
        pca_panel = upscale(rend_pcas[j], S)
        cos_panel = upscale(cosine_map_to_heatmap(cos), S)
        if pca_panel.shape[0] != tH_c or pca_panel.shape[1] != tW_c:
            pca_panel = cv2.resize(pca_panel, (tW_c, tH_c), interpolation=cv2.INTER_LINEAR)
            cos_panel = cv2.resize(cos_panel, (tW_c, tH_c), interpolation=cv2.INTER_LINEAR)

        def _to_display(img):
            """Resize any panel to composite display resolution."""
            if img.shape[0] != tH_c or img.shape[1] != tW_c:
                return cv2.resize(img, (tW_c, tH_c), interpolation=cv2.INTER_LINEAR)
            return img

        # Depth
        dpath = get_vis_depth_path(j)
        d_raw = load_vis_depth(j)
        if d_raw is not None:
            gt_d = d_raw.astype(np.float32) / 1000.0
            gt_vmin = gt_d[gt_d > 0.01].min() if (gt_d > 0.01).any() else 0
            gt_vmax = gt_d.max()
            r_depth = resize_depth_map(fused_depth_preds[j], gt_d.shape[:2])
            gt_depth_panel = _to_display(depth_to_colormap(gt_d, gt_vmin, gt_vmax))
            rend_depth_panel = _to_display(depth_to_colormap(r_depth, gt_vmin, gt_vmax))

            gd = geom_depths[j]
            has_geom = gd is not None and gd.max() > 0.01
            if has_geom:
                gd_aligned = align_depth_scale_shift(gd, gt_d)
                gd_aligned = smooth_depth_for_display(gd_aligned, gd_aligned > 0.01)
                geom_depth_panel = _to_display(
                    depth_to_colormap(gd_aligned, gt_vmin, gt_vmax))
            else:
                geom_depth_panel = np.zeros((tH_c, tW_c, 3), dtype=np.uint8)
        else:
            gt_depth_panel = np.zeros((tH_c, tW_c, 3), dtype=np.uint8)
            rend_depth_panel = np.zeros((tH_c, tW_c, 3), dtype=np.uint8)
            geom_depth_panel = np.zeros((tH_c, tW_c, 3), dtype=np.uint8)

        # Segmentation (smooth: upscale logits before argmax)
        spath = get_vis_sem_path(j)
        sem_raw_img = load_vis_sem(j)
        if sem_raw_img is not None:
            gt_sem_hr = cv2.resize(sem_raw_img, (tW_c, tH_c), interpolation=cv2.INTER_NEAREST)
            r_seg_hr = rend_seg_preds_hr[j]
            if r_seg_hr.shape[0] != tH_c or r_seg_hr.shape[1] != tW_c:
                r_seg_hr = cv2.resize(r_seg_hr, (tW_c, tH_c), interpolation=cv2.INTER_NEAREST)
            rgb_hr = cv2.resize(rgb, (tW_c, tH_c))
            gt_seg_panel = (0.6 * seg_to_color(gt_sem_hr) + 0.4 * rgb_hr).astype(np.uint8)
            rend_seg_panel = (0.6 * seg_to_color(r_seg_hr) + 0.4 * rgb_hr).astype(np.uint8)
        else:
            gt_seg_panel = np.zeros((tH_c, tW_c, 3), dtype=np.uint8)
            rend_seg_panel = np.zeros((tH_c, tW_c, 3), dtype=np.uint8)

        # Layout: 2 rows × 5 cols
        # Row 1: RGB      | Rendered PCA | GT Depth   | Geom Depth  | GT Seg
        # Row 2: Cosine   | GT PCA       | Pred Depth | Depth Error | Pred Seg
        rgb_panel = rgb_display

        gt_pca_panel = upscale(gt_pcas[j], S)
        if gt_pca_panel.shape[0] != tH_c or gt_pca_panel.shape[1] != tW_c:
            gt_pca_panel = cv2.resize(gt_pca_panel, (tW_c, tH_c), interpolation=cv2.INTER_LINEAR)

        # Depth error map (absolute difference, jet colormap)
        if d_raw is not None:
            r_depth = resize_depth_map(fused_depth_preds[j], gt_d.shape[:2])
            depth_err = np.abs(r_depth - gt_d)
            depth_err[gt_d < 0.01] = 0
            err_max = np.percentile(depth_err[gt_d > 0.01], 95) if (gt_d > 0.01).any() else 1.0
            err_norm = np.clip(depth_err / max(err_max, 1e-6), 0, 1)
            err_color = cv2.applyColorMap((err_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            err_color = cv2.cvtColor(err_color, cv2.COLOR_BGR2RGB)
            depth_err_panel = _to_display(err_color)
        else:
            depth_err_panel = np.zeros((tH_c, tW_c, 3), dtype=np.uint8)

        row1_panels = [rgb_panel, pca_panel, gt_depth_panel,
                       geom_depth_panel, gt_seg_panel]
        row1_labels = ["Input RGB", "Rendered PCA", "GT Depth (Full)",
                       "Geom Depth", "GT Segmentation"]
        row2_panels = [cos_panel, gt_pca_panel, rend_depth_panel,
                       depth_err_panel, rend_seg_panel]
        row2_labels = [f"Cosine ({cos.mean():.3f})", "GT PCA",
                       "Fused Depth", "Depth Error", "Pred Segmentation"]

        for k in range(5):
            row1_panels[k] = add_text(row1_panels[k], row1_labels[k], font_scale=0.45)
            if row2_labels[k]:
                row2_panels[k] = add_text(row2_panels[k], row2_labels[k], font_scale=0.45)

        row1 = hconcat_with_border(row1_panels, border=3)
        row2 = hconcat_with_border(row2_panels, border=3)
        composite = vconcat_with_border([row1, row2], border=3)

        save_path = dirs["composite"] / f"composite_frame_{get_vis_label(j)}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    print(f"  Saved {min(5, n_vis)} composite figures")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VISUALIZATION v2 COMPLETE")
    print("=" * 60)
    print(f"Output directory: {out_root}")
    for name, d in dirs.items():
        n_files = len(list(d.glob("*.png")))
        print(f"  {name}/: {n_files} images")

    # Compute and print quality stats
    cos_vals = []
    for j in range(n_vis):
        cos = F.cosine_similarity(
            rend_feats[j].flatten().unsqueeze(0),
            gt_feats[j].flatten().unsqueeze(0)
        ).item()
        cos_vals.append(cos)
    print(f"\nFeature quality (val novel views, {n_vis} frames):")
    print(f"  Mean cosine similarity: {np.mean(cos_vals):.4f}")
    print(f"  Min cosine similarity:  {np.min(cos_vals):.4f}")


if __name__ == "__main__":
    main()
