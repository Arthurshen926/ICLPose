"""
Evaluate RADIO-GS on downstream tasks: feature quality, depth, segmentation.

Usage:
    python radio_gs/scripts/eval_downstream.py \
        --config radio_gs/configs/replica_explicit.yaml \
        --checkpoint output/radio_gs/room0_explicit/checkpoints/best.pth \
        --tasks feature_quality depth segmentation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from radio_gs.config import RadioGSConfig, load_config
from radio_gs.data.benchmark_paths import (
    list_feature_paths,
    load_w2c_from_pose_dir,
    load_w2c_from_pose_file,
    resolve_dataset_type,
    resolve_depth_path,
    resolve_scene_root,
    resolve_semantics_path,
    resolve_split_data_dir,
    resolve_split_feature_dir,
    resolve_split_frame_ids,
    resolve_split_pose_source,
)
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer


def build_components(config: RadioGSConfig, checkpoint_path: str, device: torch.device):
    """Load trained model, codec, sharpener, renderer from checkpoint."""
    is_hybrid = getattr(config, "architecture", "explicit") == "hybrid"
    if not is_hybrid:
        from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
        latent_dim = getattr(config, "latent_dim", 64)
        model = ExplicitFeatureGaussian(latent_dim=latent_dim)
    else:
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

    ply_path = getattr(config, "ply_path", None)
    if ply_path:
        model.load_from_ply(ply_path)

    codec = HCDCodec(
        input_dim=getattr(config, "radio_feature_dim", 1280),
        bottleneck_dim=getattr(config, "bottleneck_dim", 64),
        dual_stream=getattr(config, "dual_stream", True),
    )

    sharpener = FeatSharp3D(
        mode=getattr(config, "featsharp_mode", "analytical"),
        feature_dim=latent_dim,
        strength=getattr(config, "featsharp_strength", 0.5),
    )

    renderer = FeatureFieldRenderer(
        image_height=getattr(config, "feature_height", 30),
        image_width=getattr(config, "feature_width", 40),
        fx=config.fx * config.feature_width / config.image_width,
        fy=config.fy * config.feature_height / config.image_height,
        cx=config.cx * config.feature_width / config.image_width,
        cy=config.cy * config.feature_height / config.image_height,
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=getattr(config, "use_2dgs", False),
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if "codec_state_dict" in ckpt:
        codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
    if "sharpener_state_dict" in ckpt:
        sharpener.load_state_dict(ckpt["sharpener_state_dict"], strict=False)

    model = model.to(device).eval()
    codec = codec.to(device).eval()
    sharpener = sharpener.to(device).eval()
    renderer = renderer.to(device)

    return model, codec, sharpener, renderer


@torch.no_grad()
def render_decoded(model, codec, sharpener, renderer, pose_w2c, device):
    """Render + sharpen + decode → 1280d features."""
    pose = pose_w2c.to(device)
    result = renderer.render_features(model, pose)
    compact = result["feature_map"].unsqueeze(0)  # [1, D, H, W]
    compact = sharpener(compact)
    decoded = codec.decode(compact)  # [1, 1280, H, W]
    return decoded, result


def _collect_split_inputs(config: RadioGSConfig, split: str) -> Dict[str, object]:
    feature_dir = resolve_split_feature_dir(config, split)
    frame_ids = resolve_split_frame_ids(config, split)
    feat_paths = list_feature_paths(feature_dir, frame_ids=frame_ids)
    frame_indices = [int(p.stem.split("_")[1]) for p in feat_paths]
    pose_file, pose_dir = resolve_split_pose_source(config, split)
    if pose_dir:
        w2c_poses = load_w2c_from_pose_dir(pose_dir, frame_indices)
    elif pose_file:
        w2c_poses = load_w2c_from_pose_file(pose_file, frame_indices)
    else:
        raise ValueError(f"No pose source configured for split={split}")
    return {
        "dataset_type": resolve_dataset_type(config),
        "feature_dir": feature_dir,
        "feature_paths": feat_paths,
        "frame_indices": frame_indices,
        "w2c_poses": w2c_poses,
        "depth_dir": resolve_split_data_dir(config, split, "depth"),
        "semantics_dir": resolve_split_data_dir(config, split, "semantics"),
    }


# ===================================================================
# Feature Quality Evaluation
# ===================================================================

@torch.no_grad()
def eval_feature_quality(
    model, codec, sharpener, renderer, config, device, split="val"
) -> Dict[str, float]:
    """Evaluate feature reconstruction quality vs GT RADIO features."""
    split_inputs = _collect_split_inputs(config, split)
    feat_paths = split_inputs["feature_paths"]
    w2c_poses = split_inputs["w2c_poses"]

    cosines, l2s, psnrs_norm = [], [], []

    for idx in tqdm(range(len(feat_paths)), desc=f"Feature Quality ({split})"):
        gt = torch.load(feat_paths[idx], map_location="cpu")
        if gt.dim() == 4:
            gt = gt.squeeze(0)
        gt = gt.float().unsqueeze(0).to(device)  # [1, 1280, H, W]

        pose_w2c = torch.tensor(w2c_poses[idx], dtype=torch.float32)
        decoded, _ = render_decoded(model, codec, sharpener, renderer, pose_w2c, device)

        # Match spatial dims
        if decoded.shape[-2:] != gt.shape[-2:]:
            decoded = F.interpolate(decoded, gt.shape[-2:], mode="bilinear", align_corners=False)

        # Cosine similarity (channel-wise, per pixel)
        cos = F.cosine_similarity(decoded, gt, dim=1).mean().item()
        cosines.append(cos)

        # L2
        l2 = F.mse_loss(decoded, gt).item()
        l2s.append(l2)

        # Normalized PSNR (normalize features to [0,1] range for meaningful dB)
        gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        dec_norm = (decoded - decoded.min()) / (decoded.max() - decoded.min() + 1e-8)
        mse_norm = F.mse_loss(dec_norm, gt_norm).item()
        psnr_norm = -10 * np.log10(max(mse_norm, 1e-10))
        psnrs_norm.append(psnr_norm)

    return {
        f"{split}_cosine_sim": float(np.mean(cosines)),
        f"{split}_l2_loss": float(np.mean(l2s)),
        f"{split}_psnr_norm": float(np.mean(psnrs_norm)),
        f"{split}_n_frames": len(feat_paths),
    }


# ===================================================================
# Depth Evaluation
# ===================================================================

def eval_depth(
    model, codec, sharpener, renderer, config, device, split="val"
) -> Dict[str, float]:
    """Evaluate depth prediction from decoded features with a linear probe."""
    from radio_gs.heads.depth_head import DepthHead

    eval_inputs = _collect_split_inputs(config, split)
    train_inputs = _collect_split_inputs(config, "train")
    feat_paths = eval_inputs["feature_paths"]
    frame_indices = eval_inputs["frame_indices"]
    w2c_poses = eval_inputs["w2c_poses"]
    depth_dir = eval_inputs["depth_dir"]
    dataset_type = eval_inputs["dataset_type"]

    if depth_dir is None:
        print("  No depth GT found, skipping depth eval")
        return {}

    # Train a quick linear probe on GT RADIO features → depth
    print("  Training depth linear probe...")
    feat_dim = getattr(config, "radio_feature_dim", 1280)
    depth_head = DepthHead(feat_dim, head_type="linear").to(device)
    optimizer = torch.optim.Adam(depth_head.parameters(), lr=1e-3)

    # Train on train split GT features
    train_feat_paths = train_inputs["feature_paths"]
    train_frame_indices = train_inputs["frame_indices"]
    train_depth_dir = train_inputs["depth_dir"]
    if train_depth_dir is None:
        print("  No train depth GT found, skipping depth eval")
        return {}

    train_pairs = []
    for path, frame_idx in zip(train_feat_paths, train_frame_indices):
        depth_path = resolve_depth_path(train_depth_dir, frame_idx, dataset_type)
        if depth_path is not None and depth_path.exists():
            train_pairs.append((path, depth_path))

    n_train = min(len(train_pairs), 200)
    fH, fW = getattr(config, "feature_height", 30), getattr(config, "feature_width", 40)

    depth_head.train()
    for ep in range(10):
        indices = np.random.permutation(n_train)[:50]
        for idx in indices:
            feat_path, depth_path = train_pairs[idx]
            gt_feat = torch.load(feat_path, map_location=device).float().unsqueeze(0)
            d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if d is None:
                continue
            gt_d = torch.tensor(d.astype(np.float32) / 1000.0, device=device)
            gt_d = F.interpolate(gt_d.unsqueeze(0).unsqueeze(0), (fH, fW), mode="bilinear", align_corners=False)
            pred_d = depth_head(gt_feat)  # [1, 1, fH, fW]
            valid = gt_d > 0.01
            if valid.sum() < 10:
                continue
            loss = F.l1_loss(pred_d[valid], gt_d[valid])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Now evaluate with rendered+decoded features
    depth_head.eval()
    abs_rels, rmses, delta1s = [], [], []

    for idx, frame_idx in enumerate(tqdm(frame_indices, desc="Eval Depth")):
        pose_w2c = torch.tensor(w2c_poses[idx], dtype=torch.float32)
        decoded, _ = render_decoded(model, codec, sharpener, renderer, pose_w2c, device)

        pred_d = depth_head(decoded).squeeze()  # [H, W]

        depth_path = resolve_depth_path(depth_dir, frame_idx, dataset_type)
        if depth_path is None:
            continue
        d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        gt_d = torch.tensor(d.astype(np.float32) / 1000.0, device=device)
        gt_d = F.interpolate(gt_d.unsqueeze(0).unsqueeze(0), pred_d.shape, mode="bilinear", align_corners=False).squeeze()

        valid = gt_d > 0.01
        if valid.sum() < 10:
            continue

        pred_v, gt_v = pred_d[valid], gt_d[valid]

        # Scale-invariant alignment
        scale = (gt_v * pred_v).sum() / (pred_v * pred_v).sum().clamp(min=1e-8)
        pred_v = pred_v * scale

        abs_rel = ((pred_v - gt_v).abs() / gt_v.clamp(min=1e-8)).mean().item()
        rmse = ((pred_v - gt_v) ** 2).mean().sqrt().item()
        ratio = torch.max(pred_v / gt_v.clamp(min=1e-8), gt_v / pred_v.clamp(min=1e-8))
        delta_1 = (ratio < 1.25).float().mean().item()

        abs_rels.append(abs_rel)
        rmses.append(rmse)
        delta1s.append(delta_1)

    if not abs_rels:
        return {}
    return {
        "depth_abs_rel": float(np.mean(abs_rels)),
        "depth_rmse": float(np.mean(rmses)),
        "depth_delta1": float(np.mean(delta1s)),
    }


# ===================================================================
# Segmentation Evaluation
# ===================================================================

def eval_segmentation(
    model, codec, sharpener, renderer, config, device, split="val"
) -> Dict[str, float]:
    """Evaluate semantic segmentation with a linear probe."""
    from radio_gs.heads.segmentation_head import SegmentationHead

    num_classes = getattr(config, "seg_num_classes", 40)
    eval_inputs = _collect_split_inputs(config, split)
    train_inputs = _collect_split_inputs(config, "train")
    feat_paths = eval_inputs["feature_paths"]
    frame_indices = eval_inputs["frame_indices"]
    w2c_poses = eval_inputs["w2c_poses"]
    sem_dir = eval_inputs["semantics_dir"]
    dataset_type = eval_inputs["dataset_type"]

    if sem_dir is None:
        print("  No semantic GT found, skipping segmentation eval")
        return {}

    fH, fW = getattr(config, "feature_height", 30), getattr(config, "feature_width", 40)
    feat_dim = getattr(config, "radio_feature_dim", 1280)

    # Train linear probe on GT features
    print("  Training segmentation linear probe...")
    seg_head = SegmentationHead(feat_dim, num_classes=num_classes, head_type="linear").to(device)
    optimizer = torch.optim.Adam(seg_head.parameters(), lr=1e-3)

    train_feat_paths = train_inputs["feature_paths"]
    train_frame_indices = train_inputs["frame_indices"]
    train_sem_dir = train_inputs["semantics_dir"]
    if train_sem_dir is None:
        print("  No train semantic GT found, skipping segmentation eval")
        return {}

    train_pairs = []
    for path, frame_idx in zip(train_feat_paths, train_frame_indices):
        sem_path = resolve_semantics_path(train_sem_dir, frame_idx, dataset_type)
        if sem_path is not None and sem_path.exists():
            train_pairs.append((path, sem_path))

    n_train = min(len(train_pairs), 200)

    seg_head.train()
    for ep in range(15):
        indices = np.random.permutation(n_train)[:50]
        for idx in indices:
            feat_path, sem_path = train_pairs[idx]
            gt_feat = torch.load(feat_path, map_location=device).float().unsqueeze(0)
            sem = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
            if sem is None:
                continue
            gt_sem = torch.tensor(sem.astype(np.int64), device=device)
            gt_sem = F.interpolate(gt_sem.float().unsqueeze(0).unsqueeze(0), (fH, fW), mode="nearest").squeeze().long()
            gt_sem = gt_sem.clamp(0, num_classes - 1)
            logits = seg_head(gt_feat)  # [1, C, fH, fW]
            loss = F.cross_entropy(logits, gt_sem.unsqueeze(0), ignore_index=255)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate with rendered+decoded features
    seg_head.eval()
    correct, total = 0, 0
    class_intersect = torch.zeros(num_classes, device=device)
    class_union = torch.zeros(num_classes, device=device)

    for idx, frame_idx in enumerate(tqdm(frame_indices, desc="Eval Segmentation")):
        pose_w2c = torch.tensor(w2c_poses[idx], dtype=torch.float32)
        decoded, _ = render_decoded(model, codec, sharpener, renderer, pose_w2c, device)

        logits = seg_head(decoded)  # [1, C, H, W]
        pred = logits.argmax(dim=1).squeeze(0)  # [H, W]

        sem_path = resolve_semantics_path(sem_dir, frame_idx, dataset_type)
        if sem_path is None:
            continue
        sem = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
        if sem is None:
            continue
        gt_sem = torch.tensor(sem.astype(np.int64), device=device)
        gt_sem = F.interpolate(gt_sem.float().unsqueeze(0).unsqueeze(0), pred.shape, mode="nearest").squeeze().long()
        gt_sem = gt_sem.clamp(0, num_classes - 1)

        valid = gt_sem != 255
        if valid.sum() < 10:
            continue

        correct += (pred[valid] == gt_sem[valid]).sum().item()
        total += valid.sum().item()

        for c in range(num_classes):
            pred_c = pred == c
            gt_c = gt_sem == c
            class_intersect[c] += (pred_c & gt_c & valid).sum()
            class_union[c] += ((pred_c | gt_c) & valid).sum()

    if total == 0:
        return {}

    pixel_acc = correct / total
    valid_classes = class_union > 0
    iou_per_class = class_intersect[valid_classes] / class_union[valid_classes].clamp(min=1)
    miou = iou_per_class.mean().item()

    return {
        "seg_mIoU": float(miou),
        "seg_pixel_acc": float(pixel_acc),
        "seg_num_valid_classes": int(valid_classes.sum().item()),
    }


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="RADIO-GS Downstream Evaluation")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--tasks", nargs="+",
        default=["feature_quality", "depth", "segmentation"],
        help="Tasks to evaluate",
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    output_dir = args.output_dir or os.path.join(getattr(config, "output_dir", "output"), "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model, codec, sharpener, renderer = build_components(config, args.checkpoint, device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    results = {}

    if "feature_quality" in args.tasks:
        print("\n=== Feature Reconstruction Quality ===")
        fq = eval_feature_quality(model, codec, sharpener, renderer, config, device, args.split)
        results.update(fq)
        for k, v in fq.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if "depth" in args.tasks:
        print("\n=== Depth Estimation ===")
        dm = eval_depth(model, codec, sharpener, renderer, config, device, args.split)
        results.update(dm)
        for k, v in dm.items():
            print(f"  {k}: {v:.4f}")

    if "segmentation" in args.tasks:
        print("\n=== Semantic Segmentation ===")
        sm = eval_segmentation(model, codec, sharpener, renderer, config, device, args.split)
        results.update(sm)
        for k, v in sm.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
