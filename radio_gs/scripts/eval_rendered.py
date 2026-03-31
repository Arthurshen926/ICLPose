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
    
    model = ExplicitFeatureGaussian(latent_dim=getattr(config, "latent_dim", 64))
    ply_path = getattr(config, "ply_path", "")
    if ply_path:
        model.load_from_ply(ply_path)
    model = model.to(device).eval()
    
    codec = HCDCodec(
        input_dim=getattr(config, "radio_feature_dim", 1280),
        bottleneck_dim=getattr(config, "bottleneck_dim", 64),
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
        feature_dim=getattr(config, "latent_dim", 64),
        strength=getattr(config, "featsharp_strength", 0.3),
    ).to(device).eval()
    
    # Optional screen-space refiner
    refiner = None
    if getattr(config, "use_refiner", False):
        refiner = ScreenSpaceRefiner(
            latent_dim=getattr(config, "latent_dim", 64),
            hidden_dim=getattr(config, "refiner_hidden_dim", 128),
            num_blocks=getattr(config, "refiner_num_blocks", 4),
            dropout=0.0,  # no dropout at eval
        ).to(device).eval()
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
    if "sharpener_state_dict" in ckpt:
        sharpener.load_state_dict(ckpt["sharpener_state_dict"], strict=False)
    if refiner is not None and "refiner_state_dict" in ckpt:
        refiner.load_state_dict(ckpt["refiner_state_dict"], strict=False)
    
    return model, codec, renderer, sharpener, refiner, config


def render_decoded_features(model, codec, renderer, sharpener, pose_file, refiner=None):
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
                rendered = refiner(rendered)
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
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, codec, renderer, sharpener, refiner, config = load_model_and_render(args.config, args.checkpoint)
    
    scene = getattr(config, "scene", "room_0")
    scene_root = Path("dataset") / scene
    train_split = getattr(config, "train_split", "Sequence_1")
    val_split = getattr(config, "val_split", "Sequence_2")
    
    # Subsample for speed
    train_indices = list(range(0, 900, max(1, 900 // args.n_train)))[:args.n_train]
    val_indices = list(range(0, 900, max(1, 900 // args.n_val)))[:args.n_val]
    
    print(f"\n=== Rendering features ({len(train_indices)} train, {len(val_indices)} val) ===")
    
    # Render train features
    train_poses_file = str(scene_root / train_split / "traj_w_c.txt")
    all_train_poses = np.loadtxt(train_poses_file).reshape(-1, 4, 4).astype(np.float32)
    train_w2c = np.linalg.inv(all_train_poses)
    
    train_decoded = []
    train_gt_1280 = []
    gt_dir = Path(f"output/radio_features_1280d/room_0/{train_split}/backbone")
    
    print("  Rendering train features...")
    with torch.no_grad():
        for i in tqdm(train_indices, leave=False):
            pose = torch.from_numpy(train_w2c[i:i+1]).to(device)
            result = renderer.render_features_batch(model, pose)
            rendered = sharpener(result["feature_map"])
            if refiner is not None:
                rendered = refiner(rendered)
            decoded = codec.decoder(rendered).squeeze(0).cpu()
            train_decoded.append(decoded)
            # Also load GT 1280d
            gt_feat = torch.load(gt_dir / f"rgb_{i}.pt").float()
            train_gt_1280.append(gt_feat)
    
    # Render val features
    val_poses_file = str(scene_root / val_split / "traj_w_c.txt")
    all_val_poses = np.loadtxt(val_poses_file).reshape(-1, 4, 4).astype(np.float32)
    val_w2c = np.linalg.inv(all_val_poses)
    
    gt_val_dir = Path(f"output/radio_features_1280d/room_0/{val_split}/backbone")
    val_decoded = []
    val_gt_1280 = []
    
    print("  Rendering val features...")
    with torch.no_grad():
        for i in tqdm(val_indices, leave=False):
            pose = torch.from_numpy(val_w2c[i:i+1]).to(device)
            result = renderer.render_features_batch(model, pose)
            rendered = sharpener(result["feature_map"])
            if refiner is not None:
                rendered = refiner(rendered)
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
    print("\n" + "="*70)
    print(f"{'Mode':<20} {'AbsRel':>8} {'RMSE':>8} {'δ<1.25':>8} {'mIoU':>8} {'PixAcc':>8}")
    print("-"*70)
    print(f"{'Oracle (GT)':<20} {oracle_depth['depth_abs_rel']:>8.4f} {oracle_depth['depth_rmse']:>8.4f} {oracle_depth['depth_delta1']:>8.4f} {oracle_seg['seg_mIoU']:>8.4f} {oracle_seg['seg_pixel_acc']:>8.4f}")
    print(f"{'Rendered (adapted)':<20} {rendered_depth['depth_abs_rel']:>8.4f} {rendered_depth['depth_rmse']:>8.4f} {rendered_depth['depth_delta1']:>8.4f} {rendered_seg['seg_mIoU']:>8.4f} {rendered_seg['seg_pixel_acc']:>8.4f}")
    print(f"{'Cross (GT→render)':<20} {cross_depth['depth_abs_rel']:>8.4f} {cross_depth['depth_rmse']:>8.4f} {cross_depth['depth_delta1']:>8.4f} {cross_seg['seg_mIoU']:>8.4f} {cross_seg['seg_pixel_acc']:>8.4f}")
    print("="*70)


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


if __name__ == "__main__":
    main()
