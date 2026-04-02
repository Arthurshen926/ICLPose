"""Evaluate downstream tasks using rendered features (adapted heads).

Two evaluation modes:
1. "oracle": Train+eval heads on GT RADIO 1280d features (upper bound)
2. "rendered": Train heads on DECODED rendered features, eval on val rendered (realistic)

Uses the decoder FT checkpoint which has both Gaussian features and fine-tuned decoder.
"""
import torch, sys, cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '.')
from radio_gs.config import load_config
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.screen_refiner import ScreenSpaceRefiner
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

device = torch.device("cuda")


def load_model_and_render(config_path, checkpoint_path):
    """Load trained model and render 1280d features for all frames."""
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
    ).to(device).eval()
    
    renderer = FeatureFieldRenderer(
        image_height=getattr(config, "feature_height", 30),
        image_width=getattr(config, "feature_width", 40),
        fx=getattr(config, "fx", 320.0) * getattr(config, "feature_width", 40) / getattr(config, "image_width", 640),
        fy=getattr(config, "fy", 320.0) * getattr(config, "feature_height", 30) / getattr(config, "image_height", 480),
        cx=getattr(config, "cx", 319.5) * getattr(config, "feature_width", 40) / getattr(config, "image_width", 640),
        cy=getattr(config, "cy", 239.5) * getattr(config, "feature_height", 30) / getattr(config, "image_height", 480),
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=getattr(config, "use_2dgs", False),
    ).to(device)
    
    sharpener = FeatSharp3D(
        mode=getattr(config, "featsharp_mode", "analytical"),
        feature_dim=latent_dim,
        strength=getattr(config, "featsharp_strength", 0.3),
    ).to(device).eval()
    
    # Optional screen-space refiner
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
        # Detect norm type: if checkpoint has BN keys (running_mean), use "bn"
        norm_type = getattr(config, "refiner_norm_type", "gn")
        refiner = ScreenSpaceRefiner(
            latent_dim=latent_dim,
            hidden_dim=getattr(config, "refiner_hidden_dim", 128),
            num_blocks=getattr(config, "refiner_num_blocks", 4),
            dropout=getattr(config, "refiner_dropout", 0.1),
            extra_channels=extra_ch,
            norm_type=norm_type,
        ).to(device).eval()
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
    if "sharpener_state_dict" in ckpt:
        sharpener.load_state_dict(ckpt["sharpener_state_dict"], strict=False)
    if refiner is not None and "refiner_state_dict" in ckpt:
        refiner.load_state_dict(ckpt["refiner_state_dict"], strict=False)
    
    return model, codec, renderer, sharpener, refiner, config, is_hybrid


def _load_rgb_guide(rgb_dir, idx, feature_size):
    """Load and resize an RGB image as a refiner guide tensor."""
    if rgb_dir is None:
        return None
    rgb_path = Path(rgb_dir) / f"rgb_{idx}.png"
    if not rgb_path.exists():
        return None
    img = cv2.imread(str(rgb_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (feature_size[1], feature_size[0]))
    return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def _render_rgb_guide(model, rgb_renderer, viewmat, feature_size):
    """Render RGB from 2DGS at full resolution, downsample to feature size."""
    with torch.no_grad():
        result = rgb_renderer.render_rgb(model, viewmat)
        rgb = result["rgb"].unsqueeze(0)  # [1, 3, H_full, W_full]
        rgb = F.interpolate(rgb, size=feature_size, mode="bilinear", align_corners=False)
    return rgb  # [1, 3, fH, fW]


def _render_fullres_depth(model, fullres_renderer, viewmat):
    """Render geometric depth at full image resolution from 3DGS."""
    with torch.no_grad():
        result = fullres_renderer.render_rgb(model, viewmat)
        return result["depth"]  # [H_full, W_full]


def _build_depth_guide(render_result, depth_grad=False):
    """Build depth guide (1ch or 3ch with gradients) from render result."""
    depth = render_result["depth_map"]  # [B, H, W]
    depth = depth.unsqueeze(1)           # [B, 1, H, W]
    dmin = depth.amin(dim=(2, 3), keepdim=True)
    dmax = depth.amax(dim=(2, 3), keepdim=True)
    depth = (depth - dmin) / (dmax - dmin + 1e-6)
    if depth_grad:
        dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0)) * 10.0
        dy = F.pad(dy, (0, 0, 0, 1)) * 10.0
        return torch.cat([depth, dx, dy], dim=1)  # [B, 3, H, W]
    return depth  # [B, 1, H, W]


def _hybrid_decode(model, rendered, result, pose_w2c, K):
    """Apply hybrid hash-grid decode to rendered features.
    
    Args:
        rendered: [B, latent_dim, H, W] post-sharpener/refiner latent features
        result: render result dict (contains depth_map)
        pose_w2c: [B, 4, 4] world-to-camera transform
        K: [3, 3] intrinsic matrix
    """
    from radio_gs.models.hybrid_gaussian import unproject_depth_to_positions
    depth_map = result["depth_map"].float()
    H, W = depth_map.shape[1], depth_map.shape[2]
    position_map = unproject_depth_to_positions(depth_map, pose_w2c.float(), K.float(), H, W)
    # Normalize positions to [0,1] using scene bounds
    xyz = model.get_xyz()
    margin = 0.1
    lo = xyz.min(dim=0).values - margin
    hi = xyz.max(dim=0).values + margin
    extent = (hi - lo).clamp(min=1e-6)
    position_map = ((position_map - lo.view(1, 3, 1, 1)) / extent.view(1, 3, 1, 1)).clamp(0, 1)
    return model.decode_screen_space(rendered.float(), position_map)


def render_decoded_features(model, codec, renderer, sharpener, pose_file,
                            refiner=None, rgb_dir=None, feature_size=None,
                            depth_guide=False, depth_grad=False,
                            is_hybrid=False):
    """Render and decode features for all frames."""
    poses = np.loadtxt(pose_file).reshape(-1, 4, 4).astype(np.float32)
    w2c = np.linalg.inv(poses)
    
    decoded_features = []
    with torch.no_grad():
        for i in tqdm(range(len(w2c)), desc="Rendering+Decoding", leave=False):
            pose = torch.from_numpy(w2c[i:i+1]).to(device)
            result = renderer.render_features_batch(model, pose)
            rendered = sharpener(result["feature_map"])
            if refiner is not None:
                guide = None
                if rgb_dir is not None and feature_size is not None:
                    guide = _load_rgb_guide(rgb_dir, i, feature_size)
                    if guide is not None:
                        guide = guide.to(device)
                if depth_guide:
                    dguide = _build_depth_guide(result, depth_grad)
                    if guide is not None:
                        guide = torch.cat([guide, dguide], dim=1)
                    else:
                        guide = dguide
                rendered = refiner(rendered, guide=guide)
            if is_hybrid:
                rendered = _hybrid_decode(model, rendered, result, pose, renderer.K)
            decoded = codec.decoder(rendered)  # [1, 1280, H, W]
            decoded_features.append(decoded.squeeze(0).cpu())
    return decoded_features


def eval_depth(train_features, train_depth_dir, val_features, val_depth_dir, fH=30, fW=40):
    """Train linear probe for depth and evaluate."""
    print("  Training depth probe...")
    train_X, train_Y = [], []
    for i, feat in enumerate(train_features):
        dpath = train_depth_dir / f"depth_{i}.png"
        if not dpath.exists():
            continue
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        d = torch.from_numpy(d.astype(np.float32) / 1000.0)
        d = F.interpolate(d.unsqueeze(0).unsqueeze(0), (fH, fW), mode="bilinear", align_corners=False).squeeze()
        C = feat.shape[0]
        if feat.shape[1:] != (fH, fW):
            feat = F.interpolate(feat.unsqueeze(0), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
        valid = d > 0.01
        if valid.sum() < 10:
            continue
        train_X.append(feat.reshape(C, -1).T[valid.reshape(-1)])
        train_Y.append(d.reshape(-1)[valid.reshape(-1)])
    
    train_X = torch.cat(train_X, 0).to(device)
    train_Y = torch.cat(train_Y, 0).to(device)
    
    probe = nn.Linear(train_X.shape[1], 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for ep in range(100):
        pred = probe(train_X).squeeze()
        loss = F.l1_loss(pred, train_Y)
        opt.zero_grad(); loss.backward(); opt.step()
    
    # Evaluate
    probe.eval()
    abs_rels, rmses, delta1s = [], [], []
    with torch.no_grad():
        for i, feat in enumerate(val_features):
            dpath = val_depth_dir / f"depth_{i}.png"
            if not dpath.exists():
                continue
            d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
            if d is None:
                continue
            d = torch.from_numpy(d.astype(np.float32) / 1000.0).to(device)
            d = F.interpolate(d.unsqueeze(0).unsqueeze(0), (fH, fW), mode="bilinear", align_corners=False).squeeze()
            C = feat.shape[0]
            if feat.shape[1:] != (fH, fW):
                feat_r = F.interpolate(feat.unsqueeze(0).to(device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
            else:
                feat_r = feat.to(device)
            valid = d > 0.01
            if valid.sum() < 10:
                continue
            pred = probe(feat_r.reshape(C, -1).T).squeeze().reshape(fH, fW)
            p, g = pred[valid], d[valid]
            abs_rels.append((torch.abs(p - g) / g).mean().item())
            rmses.append(torch.sqrt(((p - g)**2).mean()).item())
            delta1s.append((torch.max(p/g, g/p) < 1.25).float().mean().item())
    
    return {
        "depth_abs_rel": np.mean(abs_rels),
        "depth_rmse": np.mean(rmses),
        "depth_delta1": np.mean(delta1s),
    }


def eval_segmentation(train_features, train_sem_dir, val_features, val_sem_dir, fH=30, fW=40):
    """Train linear probe for segmentation and evaluate."""
    print("  Training segmentation probe...")
    train_X, train_Y = [], []
    for i, feat in enumerate(train_features):
        spath = train_sem_dir / f"semantic_class_{i}.png"
        if not spath.exists():
            continue
        sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        sem = torch.from_numpy(sem.astype(np.int64))
        sem = F.interpolate(sem.float().unsqueeze(0).unsqueeze(0), (fH, fW), mode="nearest").squeeze().long()
        C = feat.shape[0]
        if feat.shape[1:] != (fH, fW):
            feat = F.interpolate(feat.unsqueeze(0), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
        train_X.append(feat.reshape(C, -1).T)
        train_Y.append(sem.reshape(-1))
    
    train_X = torch.cat(train_X, 0).to(device)
    train_Y = torch.cat(train_Y, 0).to(device)
    n_classes = int(train_Y.max().item()) + 1
    
    probe = nn.Linear(train_X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for ep in range(200):
        logits = probe(train_X)
        loss = F.cross_entropy(logits, train_Y)
        opt.zero_grad(); loss.backward(); opt.step()
    
    # Evaluate
    probe.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for i, feat in enumerate(val_features):
            spath = val_sem_dir / f"semantic_class_{i}.png"
            if not spath.exists():
                continue
            sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
            if sem is None:
                continue
            sem = torch.from_numpy(sem.astype(np.int64))
            sem = F.interpolate(sem.float().unsqueeze(0).unsqueeze(0), (fH, fW), mode="nearest").squeeze().long()
            sem = sem.clamp(0, n_classes - 1)
            C = feat.shape[0]
            if feat.shape[1:] != (fH, fW):
                feat_r = F.interpolate(feat.unsqueeze(0).to(device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
            else:
                feat_r = feat.to(device)
            pred = probe(feat_r.reshape(C, -1).T).argmax(1).reshape(fH, fW).cpu()
            all_preds.append(pred.reshape(-1))
            all_gts.append(sem.reshape(-1))
    
    all_preds = torch.cat(all_preds)
    all_gts = torch.cat(all_gts)
    ious = []
    for c in range(n_classes):
        gt_c = all_gts == c
        if gt_c.sum() == 0:
            continue
        pred_c = all_preds == c
        inter = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    
    return {
        "seg_mIoU": np.mean(ious),
        "seg_pixel_acc": (all_preds == all_gts).float().mean().item(),
        "seg_n_classes": len(ious),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--use_rendered_rgb", action="store_true",
                        help="Use 2DGS-rendered RGB as refiner guide instead of GT RGB")
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, codec, renderer, sharpener, refiner, config, is_hybrid = load_model_and_render(args.config, args.checkpoint)
    
    scene = getattr(config, "scene", "room_0")
    scene_root = Path("dataset") / scene
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")
    
    # RGB guide config
    rgb_guide_enabled = getattr(config, "refiner_rgb_guide", False)
    depth_guide_enabled = getattr(config, "refiner_depth_guide", False)
    depth_grad_enabled = getattr(config, "refiner_depth_grad", False)
    self_guided = getattr(config, "self_guided", False)
    feature_size = (getattr(config, "feature_height", 30), getattr(config, "feature_width", 40))
    use_rendered_rgb = args.use_rendered_rgb or self_guided  # self_guided implies rendered RGB
    rgb_renderer = None
    if use_rendered_rgb and rgb_guide_enabled:
        # Full-resolution renderer for RGB (renders at image_width×image_height, then downsampled)
        rgb_renderer = FeatureFieldRenderer(
            image_height=getattr(config, "image_height", 480),
            image_width=getattr(config, "image_width", 640),
            fx=getattr(config, "fx", 320.0),
            fy=getattr(config, "fy", 320.0),
            cx=getattr(config, "cx", 319.5),
            cy=getattr(config, "cy", 239.5),
            use_2dgs=getattr(config, "use_2dgs", False),
        ).to(device)
        train_rgb_dir = None
        val_rgb_dir = None
    else:
        train_rgb_dir = str(scene_root / train_split / "rgb") if rgb_guide_enabled else None
        val_rgb_dir = str(scene_root / val_split / "rgb") if rgb_guide_enabled else None
    
    # Subsample for speed
    train_indices = list(range(0, 900, max(1, 900 // args.n_train)))[:args.n_train]
    val_indices = list(range(0, 900, max(1, 900 // args.n_val)))[:args.n_val]

    # Full-resolution renderer for geometric depth (renders at image resolution)
    img_h = getattr(config, "image_height", 480)
    img_w = getattr(config, "image_width", 640)
    fullres_depth_renderer = FeatureFieldRenderer(
        image_height=img_h,
        image_width=img_w,
        fx=getattr(config, "fx", 320.0),
        fy=getattr(config, "fy", 320.0),
        cx=getattr(config, "cx", 319.5),
        cy=getattr(config, "cy", 239.5),
        use_2dgs=getattr(config, "use_2dgs", False),
    ).to(device)
    
    print(f"\n=== Rendering features ({len(train_indices)} train, {len(val_indices)} val) ===")
    if rgb_guide_enabled:
        if self_guided:
            print(f"  RGB guide: SELF-RENDERED from model SH (feature_size={feature_size})")
        elif use_rendered_rgb:
            print(f"  RGB guide: RENDERED from 2DGS (feature_size={feature_size})")
        else:
            print(f"  RGB guide: GT from disk (feature_size={feature_size})")
    if depth_guide_enabled:
        print(f"  Depth guide: {'3ch (depth+grad)' if depth_grad_enabled else '1ch'}")
    
    # Render train features
    train_poses_file = str(scene_root / train_split / "traj_w_c.txt")
    all_train_poses = np.loadtxt(train_poses_file).reshape(-1, 4, 4).astype(np.float32)
    train_w2c = np.linalg.inv(all_train_poses)
    
    train_decoded = []
    train_gt_1280 = []
    train_geom_depths = []
    gt_dir = Path(f"output/radio_features_1280d/{scene}/{train_split}/backbone")
    
    print("  Rendering train features...")
    with torch.no_grad():
        for i in tqdm(train_indices, leave=False):
            pose = torch.from_numpy(train_w2c[i:i+1]).to(device)
            if self_guided and rgb_guide_enabled:
                result = renderer.render_features_and_rgb(model, pose)
                self_rgb = result["rgb"]  # [1, 3, fH, fW]
            else:
                result = renderer.render_features_batch(model, pose)
                self_rgb = None
            rendered = sharpener(result["feature_map"])
            # Geometric depth via SH-based render_rgb (feature chunk render
            # returns broken median depth for non-SH colors)
            rgb_d = renderer.render_rgb(model, torch.from_numpy(train_w2c[i]).float().to(device))
            geom_depth = rgb_d["depth"].cpu()  # [fH, fW]
            train_geom_depths.append(geom_depth)
            if refiner is not None:
                guide = None
                if self_rgb is not None:
                    guide = self_rgb
                elif rgb_renderer is not None:
                    guide = _render_rgb_guide(model, rgb_renderer, pose[0], feature_size)
                elif train_rgb_dir:
                    guide = _load_rgb_guide(train_rgb_dir, i, feature_size)
                    if guide is not None:
                        guide = guide.to(device)
                if depth_guide_enabled:
                    dguide = _build_depth_guide(result, depth_grad_enabled)
                    guide = torch.cat([guide, dguide], dim=1) if guide is not None else dguide
                rendered = refiner(rendered, guide=guide)
            if is_hybrid:
                rendered = _hybrid_decode(model, rendered, result, pose, renderer.K)
            decoded = codec.decoder(rendered).squeeze(0).cpu()
            train_decoded.append(decoded)
            # Also load GT 1280d
            gt_feat = torch.load(gt_dir / f"rgb_{i}.pt").float()
            train_gt_1280.append(gt_feat)
    
    # Render val features
    val_poses_file = str(scene_root / val_split / "traj_w_c.txt")
    all_val_poses = np.loadtxt(val_poses_file).reshape(-1, 4, 4).astype(np.float32)
    val_w2c = np.linalg.inv(all_val_poses)
    
    gt_val_dir = Path(f"output/radio_features_1280d/{scene}/{val_split}/backbone")
    val_decoded = []
    val_gt_1280 = []
    val_geom_depths = []
    val_fullres_depths = []
    
    print("  Rendering val features...")
    with torch.no_grad():
        for i in tqdm(val_indices, leave=False):
            pose = torch.from_numpy(val_w2c[i:i+1]).to(device)
            if self_guided and rgb_guide_enabled:
                result = renderer.render_features_and_rgb(model, pose)
                self_rgb = result["rgb"]
            else:
                result = renderer.render_features_batch(model, pose)
                self_rgb = None
            rendered = sharpener(result["feature_map"])
            # Geometric depth via SH-based render_rgb (correct median depth)
            rgb_d = renderer.render_rgb(model, torch.from_numpy(val_w2c[i]).float().to(device))
            geom_depth = rgb_d["depth"].cpu()  # [fH, fW]
            val_geom_depths.append(geom_depth)
            # Render full-resolution geometric depth
            fullres_d = _render_fullres_depth(model, fullres_depth_renderer, pose[0])
            val_fullres_depths.append(fullres_d.cpu())  # [img_h, img_w]
            if refiner is not None:
                guide = None
                if self_rgb is not None:
                    guide = self_rgb
                elif rgb_renderer is not None:
                    guide = _render_rgb_guide(model, rgb_renderer, pose[0], feature_size)
                elif val_rgb_dir:
                    guide = _load_rgb_guide(val_rgb_dir, i, feature_size)
                    if guide is not None:
                        guide = guide.to(device)
                if depth_guide_enabled:
                    dguide = _build_depth_guide(result, depth_grad_enabled)
                    guide = torch.cat([guide, dguide], dim=1) if guide is not None else dguide
                rendered = refiner(rendered, guide=guide)
            if is_hybrid:
                rendered = _hybrid_decode(model, rendered, result, pose, renderer.K)
            decoded = codec.decoder(rendered).squeeze(0).cpu()
            val_decoded.append(decoded)
            gt_feat = torch.load(gt_val_dir / f"rgb_{i}.pt").float()
            val_gt_1280.append(gt_feat)
    
    # Feature quality
    print("\n=== Feature Quality ===")
    cos_sims = []
    for dec, gt in zip(val_decoded, val_gt_1280):
        cos = F.cosine_similarity(dec.flatten().unsqueeze(0), gt.flatten().unsqueeze(0)).item()
        cos_sims.append(cos)
    print(f"  Val decoded cosine: {np.mean(cos_sims):.4f}")
    
    # Depth dirs
    train_depth = scene_root / train_split / "depth"
    val_depth = scene_root / val_split / "depth"
    train_sem = scene_root / train_split / "semantic_class"
    val_sem = scene_root / val_split / "semantic_class"
    
    # ====== Evaluation Mode 1: Oracle (GT features) ======
    print("\n=== ORACLE: Depth (GT features) ===")
    # Use only subsampled train GT
    train_gt_sub = [train_gt_1280[j] for j in range(len(train_indices))]
    val_gt_sub = [val_gt_1280[j] for j in range(len(val_indices))]
    # Create temporary depth/sem dirs with correct indices
    oracle_depth = eval_depth_indexed(train_gt_sub, train_indices, train_depth,
                                       val_gt_sub, val_indices, val_depth)
    print(f"  AbsRel={oracle_depth['depth_abs_rel']:.4f}  RMSE={oracle_depth['depth_rmse']:.4f}  δ<1.25={oracle_depth['depth_delta1']:.4f}")
    
    print("\n=== ORACLE: Segmentation (GT features) ===")
    oracle_seg = eval_seg_indexed(train_gt_sub, train_indices, train_sem,
                                   val_gt_sub, val_indices, val_sem)
    print(f"  mIoU={oracle_seg['seg_mIoU']:.4f}  PixelAcc={oracle_seg['seg_pixel_acc']:.4f}")
    
    # ====== Evaluation Mode 2: Rendered (adapted heads) ======
    print("\n=== RENDERED: Depth (adapted heads) ===")
    rendered_depth = eval_depth_indexed(train_decoded, train_indices, train_depth,
                                         val_decoded, val_indices, val_depth)
    print(f"  AbsRel={rendered_depth['depth_abs_rel']:.4f}  RMSE={rendered_depth['depth_rmse']:.4f}  δ<1.25={rendered_depth['depth_delta1']:.4f}")
    
    print("\n=== RENDERED: Segmentation (adapted heads) ===")
    rendered_seg = eval_seg_indexed(train_decoded, train_indices, train_sem,
                                     val_decoded, val_indices, val_sem)
    print(f"  mIoU={rendered_seg['seg_mIoU']:.4f}  PixelAcc={rendered_seg['seg_pixel_acc']:.4f}")

    # ====== Evaluation Mode 2b: Geometric depth (scale-shift aligned) ======
    print("\n=== GEOMETRIC: Depth (3DGS rendered, scale-shift aligned, 30x40) ===")
    geom_depth = eval_geom_depth(val_geom_depths, val_indices, val_depth)
    print(f"  AbsRel={geom_depth['depth_abs_rel']:.4f}  RMSE={geom_depth['depth_rmse']:.4f}  δ<1.25={geom_depth['depth_delta1']:.4f}")

    print("\n=== GEOMETRIC-HR: Depth (3DGS rendered, scale-shift aligned, full-res) ===")
    geom_hr_depth = eval_fullres_geom_depth(val_fullres_depths, val_indices, val_depth)
    print(f"  AbsRel={geom_hr_depth['depth_abs_rel']:.4f}  RMSE={geom_hr_depth['depth_rmse']:.4f}  δ<1.25={geom_hr_depth['depth_delta1']:.4f}")

    # ====== Evaluation Mode 2c: Fused depth (features + geometric) ======
    print("\n=== FUSED: Depth (features + geometric depth) ===")
    fused_depth = eval_fused_depth(train_decoded, train_geom_depths, train_indices, train_depth,
                                    val_decoded, val_geom_depths, val_indices, val_depth)
    print(f"  AbsRel={fused_depth['depth_abs_rel']:.4f}  RMSE={fused_depth['depth_rmse']:.4f}  δ<1.25={fused_depth['depth_delta1']:.4f}")
    
    # ====== Evaluation Mode 3: Cross (GT-trained heads on rendered) ======
    print("\n=== CROSS: Depth (GT-trained, rendered-eval) ===")
    cross_depth = eval_depth_indexed(train_gt_sub, train_indices, train_depth,
                                      val_decoded, val_indices, val_depth)
    print(f"  AbsRel={cross_depth['depth_abs_rel']:.4f}  RMSE={cross_depth['depth_rmse']:.4f}  δ<1.25={cross_depth['depth_delta1']:.4f}")
    
    print("\n=== CROSS: Segmentation (GT-trained, rendered-eval) ===")
    cross_seg = eval_seg_indexed(train_gt_sub, train_indices, train_sem,
                                  val_decoded, val_indices, val_sem)
    print(f"  mIoU={cross_seg['seg_mIoU']:.4f}  PixelAcc={cross_seg['seg_pixel_acc']:.4f}")
    
    # Summary table
    print("\n" + "="*90)
    print(f"{'Mode':<25} {'AbsRel':>8} {'RMSE':>8} {'δ<1.25':>8} {'mIoU':>8} {'PixAcc':>8}")
    print("-"*90)
    print(f"{'Oracle (GT feat)':<25} {oracle_depth['depth_abs_rel']:>8.4f} {oracle_depth['depth_rmse']:>8.4f} {oracle_depth['depth_delta1']:>8.4f} {oracle_seg['seg_mIoU']:>8.4f} {oracle_seg['seg_pixel_acc']:>8.4f}")
    print(f"{'Rendered (adapted)':<25} {rendered_depth['depth_abs_rel']:>8.4f} {rendered_depth['depth_rmse']:>8.4f} {rendered_depth['depth_delta1']:>8.4f} {rendered_seg['seg_mIoU']:>8.4f} {rendered_seg['seg_pixel_acc']:>8.4f}")
    print(f"{'Geom 30x40':<25} {geom_depth['depth_abs_rel']:>8.4f} {geom_depth['depth_rmse']:>8.4f} {geom_depth['depth_delta1']:>8.4f} {'   N/A':>8} {'   N/A':>8}")
    print(f"{'Geom full-res':<25} {geom_hr_depth['depth_abs_rel']:>8.4f} {geom_hr_depth['depth_rmse']:>8.4f} {geom_hr_depth['depth_delta1']:>8.4f} {'   N/A':>8} {'   N/A':>8}")
    print(f"{'Fused (feat+geom)':<25} {fused_depth['depth_abs_rel']:>8.4f} {fused_depth['depth_rmse']:>8.4f} {fused_depth['depth_delta1']:>8.4f} {'   N/A':>8} {'   N/A':>8}")
    print(f"{'Cross (GT→render)':<25} {cross_depth['depth_abs_rel']:>8.4f} {cross_depth['depth_rmse']:>8.4f} {cross_depth['depth_delta1']:>8.4f} {cross_seg['seg_mIoU']:>8.4f} {cross_seg['seg_pixel_acc']:>8.4f}")
    print("="*90)


def eval_depth_indexed(train_feats, train_idx, depth_dir, val_feats, val_idx, val_depth_dir, fH=30, fW=40):
    train_X, train_Y = [], []
    for feat, i in zip(train_feats, train_idx):
        dpath = depth_dir / f"depth_{i}.png"
        if not dpath.exists():
            continue
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        d = torch.from_numpy(d.astype(np.float32) / 1000.0)
        d = F.interpolate(d.unsqueeze(0).unsqueeze(0), (fH, fW), mode="bilinear", align_corners=False).squeeze()
        C = feat.shape[0]
        if feat.shape[1:] != (fH, fW):
            feat = F.interpolate(feat.unsqueeze(0), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
        valid = d > 0.01
        if valid.sum() < 10:
            continue
        train_X.append(feat.reshape(C, -1).T[valid.reshape(-1)])
        train_Y.append(d.reshape(-1)[valid.reshape(-1)])
    
    train_X = torch.cat(train_X, 0).to(device)
    train_Y = torch.cat(train_Y, 0).to(device)
    
    probe = nn.Linear(train_X.shape[1], 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for ep in range(100):
        pred = probe(train_X).squeeze()
        loss = F.l1_loss(pred, train_Y)
        opt.zero_grad(); loss.backward(); opt.step()
    
    probe.eval()
    abs_rels, rmses, delta1s = [], [], []
    with torch.no_grad():
        for feat, i in zip(val_feats, val_idx):
            dpath = val_depth_dir / f"depth_{i}.png"
            if not dpath.exists():
                continue
            d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
            if d is None:
                continue
            d = torch.from_numpy(d.astype(np.float32) / 1000.0).to(device)
            d = F.interpolate(d.unsqueeze(0).unsqueeze(0), (fH, fW), mode="bilinear", align_corners=False).squeeze()
            C = feat.shape[0]
            if feat.shape[1:] != (fH, fW):
                feat_r = F.interpolate(feat.unsqueeze(0).to(device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
            else:
                feat_r = feat.to(device)
            valid = d > 0.01
            if valid.sum() < 10:
                continue
            pred = probe(feat_r.reshape(C, -1).T).squeeze().reshape(fH, fW)
            p, g = pred[valid], d[valid]
            abs_rels.append((torch.abs(p - g) / g).mean().item())
            rmses.append(torch.sqrt(((p - g)**2).mean()).item())
            delta1s.append((torch.max(p/g, g/p) < 1.25).float().mean().item())
    
    return {"depth_abs_rel": np.mean(abs_rels), "depth_rmse": np.mean(rmses), "depth_delta1": np.mean(delta1s)}


def eval_seg_indexed(train_feats, train_idx, sem_dir, val_feats, val_idx, val_sem_dir, fH=30, fW=40):
    train_X, train_Y = [], []
    for feat, i in zip(train_feats, train_idx):
        spath = sem_dir / f"semantic_class_{i}.png"
        if not spath.exists():
            continue
        sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        sem = torch.from_numpy(sem.astype(np.int64))
        sem = F.interpolate(sem.float().unsqueeze(0).unsqueeze(0), (fH, fW), mode="nearest").squeeze().long()
        C = feat.shape[0]
        if feat.shape[1:] != (fH, fW):
            feat = F.interpolate(feat.unsqueeze(0), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
        train_X.append(feat.reshape(C, -1).T)
        train_Y.append(sem.reshape(-1))
    
    train_X = torch.cat(train_X, 0).to(device)
    train_Y = torch.cat(train_Y, 0).to(device)
    n_classes = int(train_Y.max().item()) + 1
    
    probe = nn.Linear(train_X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for ep in range(200):
        logits = probe(train_X)
        loss = F.cross_entropy(logits, train_Y)
        opt.zero_grad(); loss.backward(); opt.step()
    
    probe.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for feat, i in zip(val_feats, val_idx):
            spath = val_sem_dir / f"semantic_class_{i}.png"
            if not spath.exists():
                continue
            sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
            if sem is None:
                continue
            sem = torch.from_numpy(sem.astype(np.int64))
            sem = F.interpolate(sem.float().unsqueeze(0).unsqueeze(0), (fH, fW), mode="nearest").squeeze().long()
            sem = sem.clamp(0, n_classes - 1)
            C = feat.shape[0]
            if feat.shape[1:] != (fH, fW):
                feat_r = F.interpolate(feat.unsqueeze(0).to(device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
            else:
                feat_r = feat.to(device)
            pred = probe(feat_r.reshape(C, -1).T).argmax(1).reshape(fH, fW).cpu()
            all_preds.append(pred.reshape(-1))
            all_gts.append(sem.reshape(-1))
    
    all_preds = torch.cat(all_preds)
    all_gts = torch.cat(all_gts)
    ious = []
    for c in range(n_classes):
        gt_c = all_gts == c
        if gt_c.sum() == 0:
            continue
        pred_c = all_preds == c
        inter = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    
    return {"seg_mIoU": np.mean(ious), "seg_pixel_acc": (all_preds == all_gts).float().mean().item(), "seg_n_classes": len(ious)}


def eval_geom_depth(geom_depths, val_idx, val_depth_dir, fH=30, fW=40):
    """Evaluate 3DGS geometric depth directly (with scale-shift alignment)."""
    abs_rels, rmses, delta1s = [], [], []
    for geom, i in zip(geom_depths, val_idx):
        dpath = val_depth_dir / f"depth_{i}.png"
        if not dpath.exists():
            continue
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        gt = torch.from_numpy(d.astype(np.float32) / 1000.0)
        gt = F.interpolate(gt.unsqueeze(0).unsqueeze(0), (fH, fW),
                           mode="bilinear", align_corners=False).squeeze()
        valid = gt > 0.01
        if valid.sum() < 10:
            continue
        g_vals = geom[valid].float()
        gt_vals = gt[valid].float()
        # Least-squares scale-shift alignment: gt ≈ scale * geom + shift
        A = torch.stack([g_vals, torch.ones_like(g_vals)], dim=1)
        params = torch.linalg.lstsq(A, gt_vals).solution  # [scale, shift]
        aligned = geom.float() * params[0] + params[1]
        p, g = aligned[valid], gt_vals
        abs_rels.append((torch.abs(p - g) / g).mean().item())
        rmses.append(torch.sqrt(((p - g)**2).mean()).item())
        delta1s.append((torch.max(p/g, g/p) < 1.25).float().mean().item())

    return {"depth_abs_rel": np.mean(abs_rels), "depth_rmse": np.mean(rmses), "depth_delta1": np.mean(delta1s)}


def eval_fullres_geom_depth(fullres_depths, val_idx, val_depth_dir):
    """Evaluate full-resolution 3DGS geometric depth (scale-shift aligned at native image res)."""
    abs_rels, rmses, delta1s = [], [], []
    for geom, i in zip(fullres_depths, val_idx):
        dpath = val_depth_dir / f"depth_{i}.png"
        if not dpath.exists():
            continue
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        gt = torch.from_numpy(d.astype(np.float32) / 1000.0)
        H, W = gt.shape
        # Resize geom to match GT resolution if needed
        if geom.shape != gt.shape:
            geom = F.interpolate(geom.unsqueeze(0).unsqueeze(0).float(),
                                  (H, W), mode="bilinear", align_corners=False).squeeze()
        valid = gt > 0.01
        if valid.sum() < 10:
            continue
        g_vals = geom[valid].float()
        gt_vals = gt[valid].float()
        A = torch.stack([g_vals, torch.ones_like(g_vals)], dim=1)
        params = torch.linalg.lstsq(A, gt_vals).solution
        aligned = geom.float() * params[0] + params[1]
        p, g = aligned[valid], gt_vals
        abs_rels.append((torch.abs(p - g) / g).mean().item())
        rmses.append(torch.sqrt(((p - g)**2).mean()).item())
        delta1s.append((torch.max(p/g, g/p) < 1.25).float().mean().item())

    return {"depth_abs_rel": np.mean(abs_rels), "depth_rmse": np.mean(rmses), "depth_delta1": np.mean(delta1s)}


def eval_fused_depth(train_feats, train_geom, train_idx, train_depth_dir,
                     val_feats, val_geom, val_idx, val_depth_dir, fH=30, fW=40):
    """Train linear probe on features + geometric depth jointly for depth fusion."""
    print("  Training fused depth probe (features + geometric depth)...")
    train_X, train_Y = [], []
    for feat, geom, i in zip(train_feats, train_geom, train_idx):
        dpath = train_depth_dir / f"depth_{i}.png"
        if not dpath.exists():
            continue
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        d = torch.from_numpy(d.astype(np.float32) / 1000.0)
        d = F.interpolate(d.unsqueeze(0).unsqueeze(0), (fH, fW),
                           mode="bilinear", align_corners=False).squeeze()
        C = feat.shape[0]
        if feat.shape[1:] != (fH, fW):
            feat = F.interpolate(feat.unsqueeze(0), (fH, fW),
                                  mode="bilinear", align_corners=False).squeeze(0)
        valid = d > 0.01
        if valid.sum() < 10:
            continue
        # Concatenate features + geometric depth as extra channel
        geom_flat = geom.reshape(1, -1).T  # [HW, 1]
        feat_flat = feat.reshape(C, -1).T  # [HW, C]
        combined = torch.cat([feat_flat, geom_flat], dim=1)  # [HW, C+1]
        train_X.append(combined[valid.reshape(-1)])
        train_Y.append(d.reshape(-1)[valid.reshape(-1)])

    train_X = torch.cat(train_X, 0).to(device)
    train_Y = torch.cat(train_Y, 0).to(device)

    probe = nn.Linear(train_X.shape[1], 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for ep in range(100):
        pred = probe(train_X).squeeze()
        loss = F.l1_loss(pred, train_Y)
        opt.zero_grad(); loss.backward(); opt.step()

    probe.eval()
    abs_rels, rmses, delta1s = [], [], []
    with torch.no_grad():
        for feat, geom, i in zip(val_feats, val_geom, val_idx):
            dpath = val_depth_dir / f"depth_{i}.png"
            if not dpath.exists():
                continue
            d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
            if d is None:
                continue
            d = torch.from_numpy(d.astype(np.float32) / 1000.0).to(device)
            d = F.interpolate(d.unsqueeze(0).unsqueeze(0), (fH, fW),
                               mode="bilinear", align_corners=False).squeeze()
            C = feat.shape[0]
            if feat.shape[1:] != (fH, fW):
                feat_r = F.interpolate(feat.unsqueeze(0).to(device), (fH, fW),
                                        mode="bilinear", align_corners=False).squeeze(0)
            else:
                feat_r = feat.to(device)
            valid = d > 0.01
            if valid.sum() < 10:
                continue
            geom_flat = geom.reshape(1, -1).T.to(device)  # [HW, 1]
            feat_flat = feat_r.reshape(C, -1).T  # [HW, C]
            combined = torch.cat([feat_flat, geom_flat], dim=1)  # [HW, C+1]
            pred = probe(combined).squeeze().reshape(fH, fW)
            p, g = pred[valid], d[valid]
            abs_rels.append((torch.abs(p - g) / g).mean().item())
            rmses.append(torch.sqrt(((p - g)**2).mean()).item())
            delta1s.append((torch.max(p/g, g/p) < 1.25).float().mean().item())

    return {"depth_abs_rel": np.mean(abs_rels), "depth_rmse": np.mean(rmses), "depth_delta1": np.mean(delta1s)}


if __name__ == "__main__":
    main()
