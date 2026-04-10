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
from radio_gs.data.benchmark_paths import (
    list_feature_paths,
    load_w2c_from_pose_dir,
    load_w2c_from_pose_file,
    resolve_dataset_type,
    resolve_depth_path,
    resolve_rgb_path,
    resolve_scene_root,
    resolve_semantics_path,
    resolve_split_data_dir,
    resolve_split_feature_dir,
    resolve_split_frame_ids,
    resolve_split_pose_source,
)
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.depth_fusion import (
    predict_depth_fusion,
    prepare_depth_fusion_sample,
    sample_depth_fusion_training_pixels,
    train_depth_fusion_probe,
)
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.screen_refiner import (
    ScreenSpaceRefiner,
    build_depth_guide,
    build_refiner_guide,
    compute_refiner_extra_channels,
)
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

device = torch.device("cuda")


def _build_probe(in_dim, out_dim, hidden=256):
    """Build a 2-layer MLP probe (much better than linear for harder scenes)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _train_probe(probe, train_X, train_Y, epochs=300, batch_size=16384,
                 lr=1e-3, task="regression", class_weights=None):
    """Train probe with mini-batch SGD. task='regression' or 'classification'."""
    probe.to(device).train()
    if class_weights is not None:
        class_weights = class_weights.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = train_X.shape[0]
    for ep in range(epochs):
        idx = torch.randint(0, n, (min(batch_size, n),))
        batch_X = train_X[idx].to(device)
        batch_Y = train_Y[idx].to(device)
        pred = probe(batch_X)
        if task == "regression":
            loss = F.l1_loss(pred.squeeze(), batch_Y)
        else:
            loss = F.cross_entropy(pred, batch_Y, weight=class_weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
    probe.eval()
    return probe


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
        extra_ch = compute_refiner_extra_channels(
            rgb_guide=rgb_guide_enabled,
            depth_guide=depth_guide_enabled,
            depth_grad=depth_grad_enabled,
            alpha_guide=getattr(config, "refiner_alpha_guide", False),
            boundary_guide=getattr(config, "refiner_boundary_guide", False),
        )
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


def _load_rgb_guide(rgb_dir, idx, feature_size, dataset_type="replica"):
    """Load and resize an RGB image as a refiner guide tensor."""
    if rgb_dir is None:
        return None
    rgb_path = resolve_rgb_path(rgb_dir, idx, dataset_type)
    if rgb_path is None:
        return None
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


def _build_depth_guide(render_result, depth_grad=False, grad_scale=10.0):
    """Backwards-compatible wrapper for shared depth-guide logic."""
    return build_depth_guide(
        render_result["depth_map"],
        depth_grad=depth_grad,
        grad_scale=grad_scale,
    )


def _build_refiner_guide(render_result, config, rgb_guide=None):
    return build_refiner_guide(
        render_result,
        rgb_guide=rgb_guide,
        use_depth_guide=getattr(config, "refiner_depth_guide", False),
        use_depth_grad=getattr(config, "refiner_depth_grad", False),
        depth_grad_scale=getattr(config, "refiner_depth_grad_scale", 10.0),
        use_alpha_guide=getattr(config, "refiner_alpha_guide", False),
        use_boundary_guide=getattr(config, "refiner_boundary_guide", False),
    )


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
    return model.decode_screen_space(
        rendered.float(),
        position_map,
        depth_map=depth_map,
    )


def render_decoded_features(model, codec, renderer, sharpener, pose_file,
                            refiner=None, rgb_dir=None, feature_size=None,
                            depth_guide=False, depth_grad=False,
                            is_hybrid=False, dataset_type="replica", config=None):
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
                    guide = _load_rgb_guide(rgb_dir, i, feature_size, dataset_type=dataset_type)
                    if guide is not None:
                        guide = guide.to(device)
                if config is not None and (
                    depth_guide
                    or getattr(config, "refiner_alpha_guide", False)
                    or getattr(config, "refiner_boundary_guide", False)
                ):
                    guide = _build_refiner_guide(result, config, rgb_guide=guide)
                elif depth_guide:
                    dguide = _build_depth_guide(result, depth_grad, 10.0)
                    guide = torch.cat([guide, dguide], dim=1) if guide is not None else dguide
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
    
    train_X = torch.cat(train_X, 0)
    train_Y = torch.cat(train_Y, 0)
    
    probe = _build_probe(train_X.shape[1], 1)
    _train_probe(probe, train_X, train_Y, epochs=300, task="regression")
    
    # Evaluate
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
    
    train_X = torch.cat(train_X, 0)
    train_Y = torch.cat(train_Y, 0)
    
    # Remap sparse class IDs to contiguous [0, N-1]
    unique_classes = torch.unique(train_Y).tolist()
    id_to_contiguous = {c: i for i, c in enumerate(unique_classes)}
    contiguous_to_id = {i: c for c, i in id_to_contiguous.items()}
    train_Y = torch.tensor([id_to_contiguous[y.item()] for y in train_Y], dtype=torch.long)
    n_classes = len(unique_classes)
    print(f"  Seg: {n_classes} active classes (remapped from max_id={max(unique_classes)})")
    
    # Compute class weights for imbalanced datasets
    counts = torch.bincount(train_Y, minlength=n_classes).float()
    counts = counts.clamp(min=1)
    weights = (1.0 / counts)
    weights = (weights / weights.sum() * n_classes)
    
    probe = _build_probe(train_X.shape[1], n_classes)
    _train_probe(probe, train_X, train_Y, epochs=500, task="classification",
                 class_weights=weights)
    
    # Evaluate
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
            sem_remapped = torch.full_like(sem, -1)
            for orig_id, cont_id in id_to_contiguous.items():
                sem_remapped[sem == orig_id] = cont_id
            C = feat.shape[0]
            if feat.shape[1:] != (fH, fW):
                feat_r = F.interpolate(feat.unsqueeze(0).to(device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
            else:
                feat_r = feat.to(device)
            pred = probe(feat_r.reshape(C, -1).T).argmax(1).reshape(fH, fW).cpu()
            valid = sem_remapped >= 0
            all_preds.append(pred[valid].reshape(-1))
            all_gts.append(sem_remapped[valid].reshape(-1))
    
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
    
    dataset_type = resolve_dataset_type(config)
    scene = getattr(config, "scene", "room_0")
    scene_root = resolve_scene_root(config)
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
        train_rgb_root = resolve_split_data_dir(config, "train", "rgb")
        val_rgb_root = resolve_split_data_dir(config, "val", "rgb")
        train_rgb_dir = str(train_rgb_root) if rgb_guide_enabled and train_rgb_root is not None else None
        val_rgb_dir = str(val_rgb_root) if rgb_guide_enabled and val_rgb_root is not None else None
    
    # Subsample for speed
    mixed_split = getattr(config, "mixed_split", False) and dataset_type == "replica"
    if mixed_split:
        # Combine both sequences, apply same random split as training
        poses_s1 = np.loadtxt(str(scene_root / train_split / "traj_w_c.txt")).reshape(-1, 4, 4).astype(np.float32)
        poses_s2 = np.loadtxt(str(scene_root / val_split / "traj_w_c.txt")).reshape(-1, 4, 4).astype(np.float32)
        n_s1 = len(poses_s1)
        n_s2 = len(poses_s2)
        mixed_ratio = getattr(config, "mixed_train_ratio", 0.8)
        mixed_seed = getattr(config, "mixed_seed", 42)
        total = n_s1 + n_s2
        train_size = int(mixed_ratio * total)
        val_size = total - train_size
        gen = torch.Generator().manual_seed(mixed_seed)
        all_indices = list(range(total))
        perm = torch.randperm(total, generator=gen).tolist()
        train_mixed = sorted(perm[:train_size])
        val_mixed = sorted(perm[train_size:])
        
        # Subsample from mixed splits
        train_step = max(1, len(train_mixed) // args.n_train)
        train_mixed_sub = train_mixed[::train_step][:args.n_train]
        val_step = max(1, len(val_mixed) // args.n_val)
        val_mixed_sub = val_mixed[::val_step][:args.n_val]
        
        # Map combined idx → (sequence_name, frame_idx_in_sequence)
        def _split_idx(combined_idx):
            if combined_idx < n_s1:
                return train_split, combined_idx
            else:
                return val_split, combined_idx - n_s1
        
        train_seq_frame = [_split_idx(ci) for ci in train_mixed_sub]
        val_seq_frame = [_split_idx(ci) for ci in val_mixed_sub]
        
        # For rendering: need (w2c_matrix, gt_feat_dir, seq_name, frame_idx) per frame
        # Load poses from both sequences
        w2c_s1 = np.linalg.inv(poses_s1)
        w2c_s2 = np.linalg.inv(poses_s2)
        
        def _get_w2c(seq, fidx):
            return w2c_s1[fidx] if seq == train_split else w2c_s2[fidx]
        
        def _get_gt_dir(seq):
            return Path(f"output/radio_features_1280d/{scene}/{seq}/backbone")
        
        # Build dir_idx pairs for GT data loading
        train_depth_dir_idx = [(scene_root / seq / "depth", fidx) for seq, fidx in train_seq_frame]
        val_depth_dir_idx = [(scene_root / seq / "depth", fidx) for seq, fidx in val_seq_frame]
        train_sem_dir_idx = [(scene_root / seq / "semantic_class", fidx) for seq, fidx in train_seq_frame]
        val_sem_dir_idx = [(scene_root / seq / "semantic_class", fidx) for seq, fidx in val_seq_frame]
        
        # Dummy indices/dirs (not used with mixed_split, but keep API compat)
        train_indices = [fidx for _, fidx in train_seq_frame]
        val_indices = [fidx for _, fidx in val_seq_frame]
        
        print(f"\n  Mixed split: total={total}, train={len(train_mixed_sub)} (from {train_size}), "
              f"val={len(val_mixed_sub)} (from {val_size}), seed={mixed_seed}")
    else:
        def _collect_split(split_name, n_target):
            feat_dir = resolve_split_feature_dir(config, split_name)
            frame_ids = resolve_split_frame_ids(config, split_name)
            feat_paths_all = list_feature_paths(feat_dir, frame_ids=frame_ids)
            frame_ids_all = [int(p.stem.split("_")[1]) for p in feat_paths_all]
            pose_file, pose_dir = resolve_split_pose_source(config, split_name)
            if pose_dir:
                w2c_all = load_w2c_from_pose_dir(pose_dir, frame_ids_all)
            elif pose_file:
                w2c_all = load_w2c_from_pose_file(pose_file, frame_ids_all)
            else:
                raise ValueError(f"No pose source configured for split={split_name}")
            step = max(1, len(frame_ids_all) // n_target)
            sel = list(range(0, len(frame_ids_all), step))[:n_target]
            feat_root = feat_dir / "backbone" if (feat_dir / "backbone").exists() else feat_dir
            return {
                "frame_ids": [frame_ids_all[j] for j in sel],
                "w2c": w2c_all[sel],
                "feat_dir": feat_root,
                "depth_dir": resolve_split_data_dir(config, split_name, "depth"),
                "sem_dir": resolve_split_data_dir(config, split_name, "semantics"),
                "rgb_dir": resolve_split_data_dir(config, split_name, "rgb"),
            }

        train_frame_ids_cfg = resolve_split_frame_ids(config, "train")
        val_frame_ids_cfg = resolve_split_frame_ids(config, "val")
        if dataset_type != "replica" and train_frame_ids_cfg is None and val_frame_ids_cfg is None:
            feat_dir = resolve_split_feature_dir(config, "train")
            feat_paths_all = list_feature_paths(feat_dir)
            frame_ids_all = [int(p.stem.split("_")[1]) for p in feat_paths_all]
            pose_file, pose_dir = resolve_split_pose_source(config, "train")
            if pose_dir:
                w2c_all = load_w2c_from_pose_dir(pose_dir, frame_ids_all)
            elif pose_file:
                w2c_all = load_w2c_from_pose_file(pose_file, frame_ids_all)
            else:
                raise ValueError("No pose source configured for generic dataset split")
            train_ratio = getattr(config, "mixed_train_ratio", 0.8)
            seed = getattr(config, "mixed_seed", 42)
            gen = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(frame_ids_all), generator=gen).tolist()
            train_cut = int(train_ratio * len(frame_ids_all))
            train_sel = sorted(perm[:train_cut])
            val_sel = sorted(perm[train_cut:])

            def _pack(sel, n_target):
                step = max(1, len(sel) // n_target)
                sel_sub = sel[::step][:n_target]
                feat_root = feat_dir / "backbone" if (feat_dir / "backbone").exists() else feat_dir
                return {
                    "frame_ids": [frame_ids_all[j] for j in sel_sub],
                    "w2c": w2c_all[sel_sub],
                    "feat_dir": feat_root,
                    "depth_dir": resolve_split_data_dir(config, "train", "depth"),
                    "sem_dir": resolve_split_data_dir(config, "train", "semantics"),
                    "rgb_dir": resolve_split_data_dir(config, "train", "rgb"),
                }

            train_split_inputs = _pack(train_sel, args.n_train)
            val_split_inputs = _pack(val_sel, args.n_val)
        else:
            train_split_inputs = _collect_split("train", args.n_train)
            val_split_inputs = _collect_split("val", args.n_val)
        train_indices = train_split_inputs["frame_ids"]
        val_indices = val_split_inputs["frame_ids"]
        train_w2c = train_split_inputs["w2c"]
        val_w2c = val_split_inputs["w2c"]
        gt_dir = train_split_inputs["feat_dir"]
        gt_val_dir = val_split_inputs["feat_dir"]
        train_depth_dir_idx = None
        val_depth_dir_idx = None
        train_sem_dir_idx = None
        val_sem_dir_idx = None

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
    if mixed_split:
        # Mixed: each frame maps to a specific sequence
        pass  # handled below in unified loop
    else:
        train_poses_file = str(scene_root / train_split / "traj_w_c.txt")
        all_train_poses = np.loadtxt(train_poses_file).reshape(-1, 4, 4).astype(np.float32)
        train_w2c = np.linalg.inv(all_train_poses)
    
    train_decoded = []
    train_gt_1280 = []
    train_geom_depths = []
    train_geom_alphas = []
    if not mixed_split:
        gt_dir = Path(f"output/radio_features_1280d/{scene}/{train_split}/backbone")
    
    def _render_one_frame(w2c_mat, frame_idx, gt_feat_path, rgb_dir_for_guide=None):
        """Render a single frame and return decoded features plus geometric cues."""
        pose = torch.from_numpy(w2c_mat[np.newaxis]).to(device)
        if self_guided and rgb_guide_enabled:
            result = renderer.render_features_and_rgb(model, pose)
            self_rgb = result["rgb"]
            geom_depth = result.get("geom_depth")
            geom_alpha = result.get("geom_alpha")
        else:
            result = renderer.render_features_batch(model, pose)
            self_rgb = None
            geom_depth = None
            geom_alpha = None
        rendered = sharpener(result["feature_map"])
        rgb_d = None
        if geom_depth is None:
            rgb_d = renderer.render_rgb(model, torch.from_numpy(w2c_mat).float().to(device))
            geom_depth = rgb_d["depth"]
        if geom_alpha is None:
            if rgb_d is None:
                rgb_d = renderer.render_rgb(model, torch.from_numpy(w2c_mat).float().to(device))
            geom_alpha = rgb_d["alpha"]
        geom_depth = geom_depth.squeeze(0).cpu() if geom_depth.dim() == 3 else geom_depth.cpu()
        geom_alpha = geom_alpha.squeeze(0).cpu() if geom_alpha.dim() == 3 else geom_alpha.cpu()
        if refiner is not None:
            guide = None
            if self_rgb is not None:
                guide = self_rgb
            elif rgb_renderer is not None:
                guide = _render_rgb_guide(model, rgb_renderer, pose[0], feature_size)
            elif rgb_dir_for_guide:
                guide = _load_rgb_guide(rgb_dir_for_guide, frame_idx, feature_size, dataset_type=dataset_type)
                if guide is not None:
                    guide = guide.to(device)
            if depth_guide_enabled or getattr(config, "refiner_alpha_guide", False) or getattr(config, "refiner_boundary_guide", False):
                guide = _build_refiner_guide(result, config, rgb_guide=guide)
            rendered = refiner(rendered, guide=guide)
        if is_hybrid:
            rendered = _hybrid_decode(model, rendered, result, pose, renderer.K)
        decoded = codec.decoder(rendered).squeeze(0).cpu()
        gt_feat = torch.load(gt_feat_path).float()
        return decoded, gt_feat, geom_depth, geom_alpha, result, pose
    
    print("  Rendering train features...")
    with torch.no_grad():
        for j, i in enumerate(tqdm(train_indices, leave=False)):
            if mixed_split:
                seq, fidx = train_seq_frame[j]
                w2c_mat = _get_w2c(seq, fidx)
                gt_path = _get_gt_dir(seq) / f"rgb_{fidx}.pt"
                rgb_guide_dir = str(scene_root / seq / "rgb") if (rgb_guide_enabled and not use_rendered_rgb) else None
            else:
                w2c_mat = train_w2c[j]
                gt_path = gt_dir / f"rgb_{i}.pt"
                rgb_guide_dir = train_rgb_dir
            decoded, gt_feat, geom_depth, geom_alpha, _, _ = _render_one_frame(
                w2c_mat, i if not mixed_split else fidx, gt_path, rgb_guide_dir)
            train_decoded.append(decoded)
            train_gt_1280.append(gt_feat)
            train_geom_depths.append(geom_depth)
            train_geom_alphas.append(geom_alpha)
    
    # Render val features
    val_decoded = []
    val_gt_1280 = []
    val_geom_depths = []
    val_geom_alphas = []
    val_fullres_depths = []
    
    print("  Rendering val features...")
    with torch.no_grad():
        for j, i in enumerate(tqdm(val_indices, leave=False)):
            if mixed_split:
                seq, fidx = val_seq_frame[j]
                w2c_mat = _get_w2c(seq, fidx)
                gt_path = _get_gt_dir(seq) / f"rgb_{fidx}.pt"
                rgb_guide_dir = str(scene_root / seq / "rgb") if (rgb_guide_enabled and not use_rendered_rgb) else None
            else:
                w2c_mat = val_w2c[j]
                gt_path = gt_val_dir / f"rgb_{i}.pt"
                rgb_guide_dir = val_rgb_dir
            
            decoded, gt_feat, geom_depth, geom_alpha, result, pose = _render_one_frame(
                w2c_mat, i if not mixed_split else fidx, gt_path, rgb_guide_dir)
            val_decoded.append(decoded)
            val_gt_1280.append(gt_feat)
            val_geom_depths.append(geom_depth)
            val_geom_alphas.append(geom_alpha)
            # Render full-resolution geometric depth
            fullres_d = _render_fullres_depth(model, fullres_depth_renderer, pose[0])
            val_fullres_depths.append(fullres_d.cpu())
    
    # Feature quality (handle resolution mismatch by resizing decoded to GT size)
    print("\n=== Feature Quality ===")
    cos_sims = []
    for dec, gt in zip(val_decoded, val_gt_1280):
        if dec.shape != gt.shape:
            dec_resized = F.interpolate(dec.unsqueeze(0), size=gt.shape[1:], mode="bilinear", align_corners=False).squeeze(0)
        else:
            dec_resized = dec
        cos = F.cosine_similarity(dec_resized.flatten().unsqueeze(0), gt.flatten().unsqueeze(0)).item()
        cos_sims.append(cos)
    print(f"  Val decoded cosine: {np.mean(cos_sims):.4f}")
    print(f"  Rendered resolution: {val_decoded[0].shape[1:] if val_decoded else 'N/A'}")
    print(f"  GT resolution: {val_gt_1280[0].shape[1:] if val_gt_1280 else 'N/A'}")
    
    # Determine actual rendered feature resolution
    rend_fH = val_decoded[0].shape[1] if val_decoded else getattr(config, "feature_height", 30)
    rend_fW = val_decoded[0].shape[2] if val_decoded else getattr(config, "feature_width", 40)
    gt_fH = val_gt_1280[0].shape[1] if val_gt_1280 else 30
    gt_fW = val_gt_1280[0].shape[2] if val_gt_1280 else 40
    
    # Depth dirs (used as fallback when not mixed_split)
    if mixed_split:
        train_depth = scene_root / train_split / "depth"
        val_depth = scene_root / val_split / "depth"
        train_sem = scene_root / train_split / "semantic_class"
        val_sem = scene_root / val_split / "semantic_class"
    else:
        train_depth = train_split_inputs["depth_dir"]
        val_depth = val_split_inputs["depth_dir"]
        train_sem = train_split_inputs["sem_dir"]
        val_sem = val_split_inputs["sem_dir"]
    
    # ====== Evaluation Mode 1: Oracle (GT features) ======
    print("\n=== ORACLE: Depth (GT features) ===")
    train_gt_sub = [train_gt_1280[j] for j in range(len(train_indices))]
    val_gt_sub = [val_gt_1280[j] for j in range(len(val_indices))]
    oracle_depth = eval_depth_indexed(train_gt_sub, train_indices, train_depth,
                                       val_gt_sub, val_indices, val_depth,
                                       fH=gt_fH, fW=gt_fW,
                                       train_dir_idx=train_depth_dir_idx,
                                       val_dir_idx=val_depth_dir_idx,
                                       dataset_type=dataset_type)
    print(f"  AbsRel={oracle_depth['depth_abs_rel']:.4f}  RMSE={oracle_depth['depth_rmse']:.4f}  δ<1.25={oracle_depth['depth_delta1']:.4f}")
    
    print("\n=== ORACLE: Segmentation (GT features) ===")
    oracle_seg = eval_seg_indexed(train_gt_sub, train_indices, train_sem,
                                    val_gt_sub, val_indices, val_sem,
                                    fH=gt_fH, fW=gt_fW,
                                    train_dir_idx=train_sem_dir_idx,
                                    val_dir_idx=val_sem_dir_idx,
                                    dataset_type=dataset_type)
    print(f"  mIoU={oracle_seg['seg_mIoU']:.4f}  PixelAcc={oracle_seg['seg_pixel_acc']:.4f}")
    
    # ====== Evaluation Mode 2: Rendered (adapted heads) ======
    print("\n=== RENDERED: Depth (adapted heads) ===")
    rendered_depth = eval_depth_indexed(train_decoded, train_indices, train_depth,
                                          val_decoded, val_indices, val_depth,
                                          fH=rend_fH, fW=rend_fW,
                                          train_dir_idx=train_depth_dir_idx,
                                          val_dir_idx=val_depth_dir_idx,
                                          dataset_type=dataset_type)
    print(f"  AbsRel={rendered_depth['depth_abs_rel']:.4f}  RMSE={rendered_depth['depth_rmse']:.4f}  δ<1.25={rendered_depth['depth_delta1']:.4f}")
    
    print("\n=== RENDERED: Segmentation (adapted heads) ===")
    rendered_seg = eval_seg_indexed(train_decoded, train_indices, train_sem,
                                      val_decoded, val_indices, val_sem,
                                      fH=rend_fH, fW=rend_fW,
                                      train_dir_idx=train_sem_dir_idx,
                                      val_dir_idx=val_sem_dir_idx,
                                      dataset_type=dataset_type)
    print(f"  mIoU={rendered_seg['seg_mIoU']:.4f}  PixelAcc={rendered_seg['seg_pixel_acc']:.4f}")

    # ====== Evaluation Mode 2b: Geometric depth (scale-shift aligned) ======
    print("\n=== GEOMETRIC: Depth (3DGS rendered, scale-shift aligned, 30x40) ===")
    geom_depth = eval_geom_depth(val_geom_depths, val_indices, val_depth,
                                  fH=rend_fH, fW=rend_fW,
                                  val_dir_idx=val_depth_dir_idx,
                                  dataset_type=dataset_type)
    print(f"  AbsRel={geom_depth['depth_abs_rel']:.4f}  RMSE={geom_depth['depth_rmse']:.4f}  δ<1.25={geom_depth['depth_delta1']:.4f}")

    print("\n=== GEOMETRIC-HR: Depth (3DGS rendered, scale-shift aligned, full-res) ===")
    geom_hr_depth = eval_fullres_geom_depth(val_fullres_depths, val_indices, val_depth,
                                             val_dir_idx=val_depth_dir_idx,
                                             dataset_type=dataset_type)
    print(f"  AbsRel={geom_hr_depth['depth_abs_rel']:.4f}  RMSE={geom_hr_depth['depth_rmse']:.4f}  δ<1.25={geom_hr_depth['depth_delta1']:.4f}")

    # ====== Evaluation Mode 2c: Fused depth (features + geometric) ======
    print("\n=== FUSED: Depth (features + geometric depth) ===")
    fused_depth = eval_fused_depth(train_decoded, train_geom_depths, train_geom_alphas, train_indices, train_depth,
                                     val_decoded, val_geom_depths, val_geom_alphas, val_indices, val_depth,
                                     fH=rend_fH, fW=rend_fW,
                                     train_dir_idx=train_depth_dir_idx,
                                     val_dir_idx=val_depth_dir_idx,
                                     dataset_type=dataset_type)
    print(f"  AbsRel={fused_depth['depth_abs_rel']:.4f}  RMSE={fused_depth['depth_rmse']:.4f}  δ<1.25={fused_depth['depth_delta1']:.4f}")
    
    # ====== Evaluation Mode 3: Cross (GT-trained heads on rendered) ======
    print("\n=== CROSS: Depth (GT-trained, rendered-eval) ===")
    cross_depth = eval_depth_indexed(train_gt_sub, train_indices, train_depth,
                                       val_decoded, val_indices, val_depth,
                                       fH=rend_fH, fW=rend_fW,
                                       train_dir_idx=train_depth_dir_idx,
                                       val_dir_idx=val_depth_dir_idx,
                                       dataset_type=dataset_type)
    print(f"  AbsRel={cross_depth['depth_abs_rel']:.4f}  RMSE={cross_depth['depth_rmse']:.4f}  δ<1.25={cross_depth['depth_delta1']:.4f}")
    
    print("\n=== CROSS: Segmentation (GT-trained, rendered-eval) ===")
    cross_seg = eval_seg_indexed(train_gt_sub, train_indices, train_sem,
                                   val_decoded, val_indices, val_sem,
                                   fH=rend_fH, fW=rend_fW,
                                   train_dir_idx=train_sem_dir_idx,
                                   val_dir_idx=val_sem_dir_idx,
                                   dataset_type=dataset_type)
    print(f"  mIoU={cross_seg['seg_mIoU']:.4f}  PixelAcc={cross_seg['seg_pixel_acc']:.4f}")

    # ====== Seg-based Grounding (replaces broken SigLIP2 projection) ======
    print("\n=== ORACLE: Seg-Grounding (GT features) ===")
    oracle_grnd = eval_seg_grounding(train_gt_sub, train_indices, train_sem,
                                     val_gt_sub, val_indices, val_sem,
                                     scene=scene, fH=gt_fH, fW=gt_fW,
                                     train_dir_idx=train_sem_dir_idx,
                                     val_dir_idx=val_sem_dir_idx,
                                     dataset_type=dataset_type)

    print("\n=== RENDERED: Seg-Grounding (adapted heads) ===")
    rendered_grnd = eval_seg_grounding(train_decoded, train_indices, train_sem,
                                       val_decoded, val_indices, val_sem,
                                       scene=scene, fH=rend_fH, fW=rend_fW,
                                       train_dir_idx=train_sem_dir_idx,
                                       val_dir_idx=val_sem_dir_idx,
                                       dataset_type=dataset_type)

    print("\n=== CROSS: Seg-Grounding (GT-trained, rendered-eval) ===")
    cross_grnd = eval_seg_grounding(train_gt_sub, train_indices, train_sem,
                                    val_decoded, val_indices, val_sem,
                                    scene=scene, fH=rend_fH, fW=rend_fW,
                                    train_dir_idx=train_sem_dir_idx,
                                    val_dir_idx=val_sem_dir_idx,
                                    dataset_type=dataset_type)

    # Helper to safely extract grounding metrics
    def _g(d, key, default="   N/A"):
        return f"{d[key]:>8.4f}" if d and key in d else f"{default:>8}"

    # Summary table
    print("\n" + "="*114)
    print(f"{'Mode':<25} {'AbsRel':>8} {'RMSE':>8} {'δ<1.25':>8} {'mIoU':>8} {'PixAcc':>8} {'Grnd_mAP':>9} {'Grnd_IoU':>9} {'Grnd_Cor':>9}")
    print("-"*114)
    print(f"{'Oracle (GT feat)':<25} {oracle_depth['depth_abs_rel']:>8.4f} {oracle_depth['depth_rmse']:>8.4f} {oracle_depth['depth_delta1']:>8.4f} {oracle_seg['seg_mIoU']:>8.4f} {oracle_seg['seg_pixel_acc']:>8.4f} {_g(oracle_grnd, 'grnd_mAP')} {_g(oracle_grnd, 'grnd_mIoU@0.5')} {_g(oracle_grnd, 'grnd_corr')}")
    print(f"{'Rendered (adapted)':<25} {rendered_depth['depth_abs_rel']:>8.4f} {rendered_depth['depth_rmse']:>8.4f} {rendered_depth['depth_delta1']:>8.4f} {rendered_seg['seg_mIoU']:>8.4f} {rendered_seg['seg_pixel_acc']:>8.4f} {_g(rendered_grnd, 'grnd_mAP')} {_g(rendered_grnd, 'grnd_mIoU@0.5')} {_g(rendered_grnd, 'grnd_corr')}")
    print(f"{'Geom 30x40':<25} {geom_depth['depth_abs_rel']:>8.4f} {geom_depth['depth_rmse']:>8.4f} {geom_depth['depth_delta1']:>8.4f} {'   N/A':>8} {'   N/A':>8} {'   N/A':>9} {'   N/A':>9} {'   N/A':>9}")
    print(f"{'Geom full-res':<25} {geom_hr_depth['depth_abs_rel']:>8.4f} {geom_hr_depth['depth_rmse']:>8.4f} {geom_hr_depth['depth_delta1']:>8.4f} {'   N/A':>8} {'   N/A':>8} {'   N/A':>9} {'   N/A':>9} {'   N/A':>9}")
    print(f"{'Fused (feat+geom)':<25} {fused_depth['depth_abs_rel']:>8.4f} {fused_depth['depth_rmse']:>8.4f} {fused_depth['depth_delta1']:>8.4f} {'   N/A':>8} {'   N/A':>8} {'   N/A':>9} {'   N/A':>9} {'   N/A':>9}")
    print(f"{'Cross (GT→render)':<25} {cross_depth['depth_abs_rel']:>8.4f} {cross_depth['depth_rmse']:>8.4f} {cross_depth['depth_delta1']:>8.4f} {cross_seg['seg_mIoU']:>8.4f} {cross_seg['seg_pixel_acc']:>8.4f} {_g(cross_grnd, 'grnd_mAP')} {_g(cross_grnd, 'grnd_mIoU@0.5')} {_g(cross_grnd, 'grnd_corr')}")
    print("="*114)


def eval_depth_indexed(train_feats, train_idx, depth_dir, val_feats, val_idx, val_depth_dir, fH=30, fW=40,
                       train_dir_idx=None, val_dir_idx=None, dataset_type="replica"):
    """Train linear probe on features for depth and evaluate.
    
    If train_dir_idx / val_dir_idx are provided (lists of (dir_path, frame_idx) tuples),
    use them for loading depth GT; otherwise use depth_dir/val_depth_dir + train_idx/val_idx.
    """
    _train_pairs = train_dir_idx if train_dir_idx else [(depth_dir, i) for i in train_idx]
    _val_pairs = val_dir_idx if val_dir_idx else [(val_depth_dir, i) for i in val_idx]

    train_X, train_Y = [], []
    for feat, (ddir, i) in zip(train_feats, _train_pairs):
        dpath = resolve_depth_path(ddir, i, dataset_type)
        if dpath is None:
            continue
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
    
    train_X = torch.cat(train_X, 0)
    train_Y = torch.cat(train_Y, 0)
    
    probe = _build_probe(train_X.shape[1], 1)
    _train_probe(probe, train_X, train_Y, epochs=300, task="regression")
    
    abs_rels, rmses, delta1s = [], [], []
    with torch.no_grad():
        for feat, (ddir, i) in zip(val_feats, _val_pairs):
            dpath = resolve_depth_path(ddir, i, dataset_type)
            if dpath is None:
                continue
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


def eval_seg_indexed(train_feats, train_idx, sem_dir, val_feats, val_idx, val_sem_dir, fH=30, fW=40,
                     train_dir_idx=None, val_dir_idx=None, dataset_type="replica"):
    """Train linear probe for segmentation and evaluate.
    
    If train_dir_idx / val_dir_idx are provided (lists of (dir_path, frame_idx) tuples),
    use them for loading semantic GT; otherwise use sem_dir/val_sem_dir + train_idx/val_idx.
    """
    _train_pairs = train_dir_idx if train_dir_idx else [(sem_dir, i) for i in train_idx]
    _val_pairs = val_dir_idx if val_dir_idx else [(val_sem_dir, i) for i in val_idx]

    train_X, train_Y = [], []
    for feat, (sdir, i) in zip(train_feats, _train_pairs):
        spath = resolve_semantics_path(sdir, i, dataset_type)
        if spath is None:
            continue
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
    
    train_X = torch.cat(train_X, 0)
    train_Y = torch.cat(train_Y, 0)
    
    # Remap sparse class IDs to contiguous [0, N-1]
    unique_classes = torch.unique(train_Y).tolist()
    id_to_contiguous = {c: i for i, c in enumerate(unique_classes)}
    train_Y = torch.tensor([id_to_contiguous[y.item()] for y in train_Y], dtype=torch.long)
    n_classes = len(unique_classes)
    print(f"  Seg: {n_classes} active classes (remapped from max_id={max(unique_classes)})")
    
    counts = torch.bincount(train_Y, minlength=n_classes).float().clamp(min=1)
    weights = (1.0 / counts)
    weights = (weights / weights.sum() * n_classes)
    
    probe = _build_probe(train_X.shape[1], n_classes)
    _train_probe(probe, train_X, train_Y, epochs=500, task="classification",
                 class_weights=weights)
    
    all_preds, all_gts = [], []
    with torch.no_grad():
        for feat, (sdir, i) in zip(val_feats, _val_pairs):
            spath = resolve_semantics_path(sdir, i, dataset_type)
            if spath is None:
                continue
            if not spath.exists():
                continue
            sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
            if sem is None:
                continue
            sem = torch.from_numpy(sem.astype(np.int64))
            sem = F.interpolate(sem.float().unsqueeze(0).unsqueeze(0), (fH, fW), mode="nearest").squeeze().long()
            sem_remapped = torch.full_like(sem, -1)
            for orig_id, cont_id in id_to_contiguous.items():
                sem_remapped[sem == orig_id] = cont_id
            C = feat.shape[0]
            if feat.shape[1:] != (fH, fW):
                feat_r = F.interpolate(feat.unsqueeze(0).to(device), (fH, fW), mode="bilinear", align_corners=False).squeeze(0)
            else:
                feat_r = feat.to(device)
            pred = probe(feat_r.reshape(C, -1).T).argmax(1).reshape(fH, fW).cpu()
            valid = sem_remapped >= 0
            all_preds.append(pred[valid].reshape(-1))
            all_gts.append(sem_remapped[valid].reshape(-1))
    
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


def eval_seg_grounding(train_feats, train_idx, sem_dir, val_feats, val_idx, val_sem_dir,
                       scene="room_0", fH=30, fW=40,
                       train_dir_idx=None, val_dir_idx=None, dataset_type="replica"):
    """Evaluate text grounding using segmentation probe softmax as heatmap.

    Instead of projecting to SigLIP2 space and computing cosine similarity
    with text embeddings (which gives near-random results), this function
    trains a per-pixel linear segmentation probe and uses the softmax
    probability for each grounding query's class as the grounding heatmap.

    Metrics per query:
      - IoU@0.5: threshold softmax probability at 0.5 → binary mask vs GT
      - mAP: average precision using softmax probability as confidence score
      - Heatmap correlation: Pearson correlation between softmax heatmap and GT binary mask
    """
    if resolve_dataset_type(dataset_type) == "scannet":
        from radio_gs.scannet_constants import GROUNDING_QUERIES
        room_ids = set()
    else:
        from radio_gs.replica_constants import GROUNDING_QUERIES, ROOM_CLASS_IDS
        room_ids = ROOM_CLASS_IDS.get(scene, set())

    # ── Train segmentation probe (same as eval_seg_indexed) ──────────────
    _train_pairs = train_dir_idx if train_dir_idx else [(sem_dir, i) for i in train_idx]
    _val_pairs = val_dir_idx if val_dir_idx else [(val_sem_dir, i) for i in val_idx]

    train_X, train_Y = [], []
    for feat, (sdir, i) in zip(train_feats, _train_pairs):
        spath = resolve_semantics_path(sdir, i, dataset_type)
        if spath is None:
            continue
        if not spath.exists():
            continue
        sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        sem = torch.from_numpy(sem.astype(np.int64))
        sem = F.interpolate(sem.float().unsqueeze(0).unsqueeze(0),
                            (fH, fW), mode="nearest").squeeze().long()
        C = feat.shape[0]
        if feat.shape[1:] != (fH, fW):
            feat = F.interpolate(feat.unsqueeze(0), (fH, fW),
                                 mode="bilinear", align_corners=False).squeeze(0)
        train_X.append(feat.reshape(C, -1).T)
        train_Y.append(sem.reshape(-1))

    train_X = torch.cat(train_X, 0)
    train_Y = torch.cat(train_Y, 0)

    unique_classes = torch.unique(train_Y).tolist()
    id_to_contiguous = {c: i for i, c in enumerate(unique_classes)}
    contiguous_to_id = {i: c for c, i in id_to_contiguous.items()}
    train_Y_mapped = torch.tensor([id_to_contiguous[y.item()] for y in train_Y], dtype=torch.long)
    n_classes = len(unique_classes)

    counts = torch.bincount(train_Y_mapped, minlength=n_classes).float().clamp(min=1)
    weights = (1.0 / counts)
    weights = (weights / weights.sum() * n_classes)

    probe = _build_probe(train_X.shape[1], n_classes)
    _train_probe(probe, train_X, train_Y_mapped, epochs=500, task="classification",
                 class_weights=weights)

    # ── Determine active grounding queries for this scene ────────────────
    active_queries = {}
    for qname, cid in sorted(GROUNDING_QUERIES.items(), key=lambda x: x[1]):
        if cid in id_to_contiguous and (not room_ids or cid in room_ids):
            active_queries[qname] = cid
    if not active_queries:
        print("  No grounding queries matched scene classes — skipping.")
        return {}

    print(f"  Seg-grounding: {len(active_queries)} queries for {scene}: "
          f"{list(active_queries.keys())}")

    # ── Evaluate on validation set ───────────────────────────────────────
    per_query_iou = {q: [] for q in active_queries}
    per_query_ap = {q: [] for q in active_queries}
    per_query_corr = {q: [] for q in active_queries}

    with torch.no_grad():
        for feat, (sdir, i) in zip(val_feats, _val_pairs):
            spath = resolve_semantics_path(sdir, i, dataset_type)
            if spath is None:
                continue
            if not spath.exists():
                continue
            sem = cv2.imread(str(spath), cv2.IMREAD_GRAYSCALE)
            if sem is None:
                continue
            sem = torch.from_numpy(sem.astype(np.int64))
            sem = F.interpolate(sem.float().unsqueeze(0).unsqueeze(0),
                                (fH, fW), mode="nearest").squeeze().long()

            C = feat.shape[0]
            if feat.shape[1:] != (fH, fW):
                feat_r = F.interpolate(feat.unsqueeze(0).to(device), (fH, fW),
                                       mode="bilinear", align_corners=False).squeeze(0)
            else:
                feat_r = feat.to(device)

            # Get softmax probabilities over all classes
            logits = probe(feat_r.reshape(C, -1).T)          # [H*W, n_classes]
            probs = F.softmax(logits, dim=1)                  # [H*W, n_classes]
            probs_2d = probs.T.reshape(n_classes, fH, fW)     # [n_classes, H, W]

            for qname, cid in active_queries.items():
                gt_mask = (sem == cid)
                if gt_mask.sum() == 0:
                    continue

                cont_id = id_to_contiguous[cid]
                heatmap = probs_2d[cont_id]                   # [H, W] on device
                gt_mask_dev = gt_mask.to(device)
                gt_binary = gt_mask_dev.float()

                # IoU @ 0.5
                pred_mask = heatmap > 0.5
                inter = (pred_mask & gt_mask_dev).float().sum()
                union = (pred_mask | gt_mask_dev).float().sum()
                iou = (inter / union).item() if union > 0 else 0.0
                per_query_iou[qname].append(iou)

                # Average Precision
                scores = heatmap.flatten().float()
                labels = gt_binary.flatten()
                sorted_idx = scores.argsort(descending=True)
                labels_sorted = labels[sorted_idx]
                tp_cum = labels_sorted.cumsum(0)
                precision = tp_cum / torch.arange(
                    1, len(labels_sorted) + 1, device=device, dtype=torch.float32)
                ap = (precision * labels_sorted).sum() / labels_sorted.sum()
                per_query_ap[qname].append(ap.item())

                # Pearson correlation between heatmap and GT binary mask
                h = heatmap.flatten().float()
                g = gt_binary.flatten()
                h_centered = h - h.mean()
                g_centered = g - g.mean()
                denom = h_centered.norm() * g_centered.norm()
                corr = (h_centered @ g_centered / denom).item() if denom > 0 else 0.0
                per_query_corr[qname].append(corr)

    # ── Aggregate and print ──────────────────────────────────────────────
    print(f"\n  {'Query':<18} {'IoU@0.5':>8} {'mAP':>8} {'Corr':>8} {'#frames':>8}")
    print(f"  {'-'*54}")
    all_ious, all_aps, all_corrs = [], [], []
    for qname in active_queries:
        if not per_query_iou[qname]:
            continue
        m_iou = np.mean(per_query_iou[qname])
        m_ap = np.mean(per_query_ap[qname])
        m_corr = np.mean(per_query_corr[qname])
        n_frames = len(per_query_iou[qname])
        print(f"  {qname:<18} {m_iou:>8.4f} {m_ap:>8.4f} {m_corr:>8.4f} {n_frames:>8d}")
        all_ious.append(m_iou)
        all_aps.append(m_ap)
        all_corrs.append(m_corr)

    mean_iou = np.mean(all_ious) if all_ious else 0.0
    mean_ap = np.mean(all_aps) if all_aps else 0.0
    mean_corr = np.mean(all_corrs) if all_corrs else 0.0
    print(f"  {'-'*54}")
    print(f"  {'MEAN':<18} {mean_iou:>8.4f} {mean_ap:>8.4f} {mean_corr:>8.4f}")

    return {
        "grnd_mIoU@0.5": mean_iou,
        "grnd_mAP": mean_ap,
        "grnd_corr": mean_corr,
        "grnd_n_queries": len(all_ious),
    }


def eval_geom_depth(geom_depths, val_idx, val_depth_dir, fH=30, fW=40, val_dir_idx=None, dataset_type="replica"):
    """Evaluate 3DGS geometric depth directly (with scale-shift alignment)."""
    _val_pairs = val_dir_idx if val_dir_idx else [(val_depth_dir, i) for i in val_idx]
    abs_rels, rmses, delta1s = [], [], []
    for geom, (ddir, i) in zip(geom_depths, _val_pairs):
        dpath = resolve_depth_path(ddir, i, dataset_type)
        if dpath is None:
            continue
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


def eval_fullres_geom_depth(fullres_depths, val_idx, val_depth_dir, val_dir_idx=None, dataset_type="replica"):
    """Evaluate full-resolution 3DGS geometric depth (scale-shift aligned at native image res)."""
    _val_pairs = val_dir_idx if val_dir_idx else [(val_depth_dir, i) for i in val_idx]
    abs_rels, rmses, delta1s = [], [], []
    for geom, (ddir, i) in zip(fullres_depths, _val_pairs):
        dpath = resolve_depth_path(ddir, i, dataset_type)
        if dpath is None:
            continue
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


def eval_fused_depth(train_feats, train_geom, train_alpha, train_idx, train_depth_dir,
                     val_feats, val_geom, val_alpha, val_idx, val_depth_dir, fH=30, fW=40,
                     train_dir_idx=None, val_dir_idx=None, dataset_type="replica"):
    """Train a learned depth-fusion probe on feature and geometric depth cues."""
    print("  Training fused depth probe (aligned geom + learned gating)...")
    _train_pairs = train_dir_idx if train_dir_idx else [(train_depth_dir, i) for i in train_idx]
    _val_pairs = val_dir_idx if val_dir_idx else [(val_depth_dir, i) for i in val_idx]

    depth_train_X, depth_train_Y = [], []
    for feat, (ddir, i) in zip(train_feats, _train_pairs):
        dpath = resolve_depth_path(ddir, i, dataset_type)
        if dpath is None:
            continue
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
        depth_train_X.append(feat.reshape(C, -1).T[valid.reshape(-1)])
        depth_train_Y.append(d.reshape(-1)[valid.reshape(-1)])

    depth_train_X = torch.cat(depth_train_X, 0)
    depth_train_Y = torch.cat(depth_train_Y, 0)
    depth_probe = _build_probe(depth_train_X.shape[1], 1)
    _train_probe(depth_probe, depth_train_X, depth_train_Y, epochs=300, task="regression")

    train_input, train_feat_depth, train_geom_depth, train_geom_valid, train_Y = [], [], [], [], []
    fusion_train_pixel_budget = 100_000
    per_frame_budget = max(256, fusion_train_pixel_budget // max(1, len(_train_pairs)))
    fusion_sample_gen = torch.Generator(device="cpu").manual_seed(42)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for feat, geom, alpha, (ddir, i) in zip(train_feats, train_geom, train_alpha, _train_pairs):
        dpath = resolve_depth_path(ddir, i, dataset_type)
        if dpath is None:
            continue
        if not dpath.exists():
            continue
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        d = torch.from_numpy(d.astype(np.float32) / 1000.0)
        d = F.interpolate(d.unsqueeze(0).unsqueeze(0), (fH, fW),
                           mode="bilinear", align_corners=False).squeeze()
        sample = prepare_depth_fusion_sample(
            feat, geom, alpha, depth_probe, fH, fW, device, output_device="cpu"
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

    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not train_input:
        raise RuntimeError("No valid samples collected for fused depth probe training.")
    fusion_probe = train_depth_fusion_probe(
        torch.cat(train_input, 0),
        torch.cat(train_feat_depth, 0),
        torch.cat(train_geom_depth, 0),
        torch.cat(train_geom_valid, 0),
        torch.cat(train_Y, 0),
        device,
        epochs=300,
    )

    abs_rels, rmses, delta1s = [], [], []
    with torch.no_grad():
        for feat, geom, alpha, (ddir, i) in zip(val_feats, val_geom, val_alpha, _val_pairs):
            dpath = resolve_depth_path(ddir, i, dataset_type)
            if dpath is None:
                continue
            if not dpath.exists():
                continue
            d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
            if d is None:
                continue
            d = torch.from_numpy(d.astype(np.float32) / 1000.0).to(device)
            d = F.interpolate(d.unsqueeze(0).unsqueeze(0), (fH, fW),
                               mode="bilinear", align_corners=False).squeeze()
            valid = d > 0.01
            if valid.sum() < 10:
                continue
            sample = prepare_depth_fusion_sample(feat, geom, alpha, depth_probe, fH, fW, device)
            pred = predict_depth_fusion(fusion_probe, sample, fH, fW)["depth"]
            p, g = pred[valid], d[valid]
            abs_rels.append((torch.abs(p - g) / g).mean().item())
            rmses.append(torch.sqrt(((p - g)**2).mean()).item())
            delta1s.append((torch.max(p/g, g/p) < 1.25).float().mean().item())

    return {"depth_abs_rel": np.mean(abs_rels), "depth_rmse": np.mean(rmses), "depth_delta1": np.mean(delta1s)}


if __name__ == "__main__":
    main()
