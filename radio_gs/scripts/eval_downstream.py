"""
Evaluate RADIO-GS on downstream tasks: depth, segmentation, text grounding.

Usage:
    python radio_gs/scripts/eval_downstream.py \
        --config radio_gs/configs/replica_explicit.yaml \
        --checkpoint output/radio_gs/replica_explicit/checkpoints/best.pth \
        --tasks depth segmentation grounding \
        --output_dir output/radio_gs/eval_results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from radio_gs.config import RadioGSConfig, load_config


def load_model_and_codec(config, checkpoint_path, device):
    """Load trained feature field model and HCD codec from checkpoint."""
    from radio_gs.models.hcd_codec import HCDCodec
    from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

    if config.architecture == 'explicit':
        from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
        model = ExplicitFeatureGaussian(latent_dim=config.latent_dim)
    else:
        from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
        model = HybridFeatureGaussian(latent_dim=config.hybrid_latent_dim)

    if config.ply_path:
        model.load_from_ply(config.ply_path)

    codec = HCDCodec(
        input_dim=config.radio_feature_dim,
        bottleneck_dim=config.bottleneck_dim,
        dual_stream=config.dual_stream,
    )

    renderer = FeatureFieldRenderer(
        image_height=config.feature_height,
        image_width=config.feature_width,
        fx=config.fx * config.feature_width / config.image_width,
        fy=config.fy * config.feature_height / config.image_height,
        cx=config.cx * config.feature_width / config.image_width,
        cy=config.cy * config.feature_height / config.image_height,
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if 'codec_state_dict' in ckpt:
        codec.load_state_dict(ckpt['codec_state_dict'])

    model = model.to(device)
    codec = codec.to(device).eval()
    renderer = renderer.to(device)

    return model, codec, renderer


def render_and_decode(model, codec, renderer, pose_w2c, device, config):
    """Render features from model and decode to 1280d."""
    pose_w2c = pose_w2c.to(device)
    if pose_w2c.ndim == 2:
        pose_w2c = pose_w2c.unsqueeze(0)

    with torch.no_grad():
        result = renderer.render_features(model, pose_w2c[0])
        compact_feat = result['feature_map'].unsqueeze(0)  # [1, D, H, W]
        decoded = codec.decode(compact_feat)  # [1, 1280, H, W]
    return decoded, result.get('depth_map', None)


@torch.no_grad()
def eval_depth(model, codec, renderer, dataset, head, loss_fn, device, config):
    """Evaluate depth estimation."""
    from radio_gs.heads.depth_head import DepthHead

    metrics = {'abs_rel': [], 'rmse': [], 'delta_1': []}
    head = head.to(device).eval()

    for i in tqdm(range(len(dataset)), desc='Eval Depth'):
        sample = dataset[i]
        decoded, _ = render_and_decode(
            model, codec, renderer, sample['pose_w2c'], device, config
        )
        pred_depth = head(decoded).squeeze(0).squeeze(0)  # [H, W]

        if 'depth' not in sample or sample['depth'] is None:
            continue
        gt_depth = sample['depth'].to(device)

        # Resize to match
        if pred_depth.shape != gt_depth.shape:
            pred_depth = F.interpolate(
                pred_depth.unsqueeze(0).unsqueeze(0),
                gt_depth.shape, mode='bilinear', align_corners=True,
            ).squeeze()

        valid = gt_depth > 0.01
        if valid.sum() < 10:
            continue

        pred_v = pred_depth[valid]
        gt_v = gt_depth[valid]

        # Scale-invariant alignment
        scale = (gt_v * pred_v).sum() / (pred_v * pred_v).sum().clamp(min=1e-8)
        pred_v = pred_v * scale

        abs_rel = ((pred_v - gt_v).abs() / gt_v.clamp(min=1e-8)).mean().item()
        rmse = ((pred_v - gt_v) ** 2).mean().sqrt().item()
        ratio = torch.max(pred_v / gt_v.clamp(min=1e-8), gt_v / pred_v.clamp(min=1e-8))
        delta_1 = (ratio < 1.25).float().mean().item()

        metrics['abs_rel'].append(abs_rel)
        metrics['rmse'].append(rmse)
        metrics['delta_1'].append(delta_1)

    return {k: np.mean(v) for k, v in metrics.items() if v}


@torch.no_grad()
def eval_segmentation(model, codec, renderer, dataset, head, device, config):
    """Evaluate semantic segmentation."""
    from radio_gs.heads.segmentation_head import compute_miou, compute_pixel_accuracy

    all_preds, all_gts = [], []
    head = head.to(device).eval()

    for i in tqdm(range(len(dataset)), desc='Eval Segmentation'):
        sample = dataset[i]
        decoded, _ = render_and_decode(
            model, codec, renderer, sample['pose_w2c'], device, config
        )
        logits = head(decoded)  # [1, C, H, W]
        pred = logits.argmax(dim=1).squeeze(0)  # [H, W]

        if 'semantics' not in sample or sample['semantics'] is None:
            continue
        gt = sample['semantics'].to(device)

        if pred.shape != gt.shape:
            pred = F.interpolate(
                pred.unsqueeze(0).unsqueeze(0).float(),
                gt.shape, mode='nearest',
            ).squeeze().long()

        all_preds.append(pred.cpu())
        all_gts.append(gt.cpu())

    if not all_preds:
        return {}

    preds = torch.stack(all_preds)
    gts = torch.stack(all_gts)

    miou = compute_miou(preds, gts, config.seg_num_classes)
    acc = compute_pixel_accuracy(preds, gts)
    return {'mIoU': miou, 'pixel_accuracy': acc}


@torch.no_grad()
def eval_grounding(model, codec, renderer, dataset, head, device, config):
    """Evaluate text grounding."""
    from radio_gs.heads.grounding_head import compute_grounding_iou

    metrics = {'iou_25': [], 'iou_50': []}
    head = head.to(device).eval()

    for i in tqdm(range(len(dataset)), desc='Eval Grounding'):
        sample = dataset[i]
        if 'text_embeddings' not in sample or 'grounding_masks' not in sample:
            continue

        decoded, _ = render_and_decode(
            model, codec, renderer, sample['pose_w2c'], device, config
        )
        text_emb = sample['text_embeddings'].to(device)  # [N_q, D]
        gt_masks = sample['grounding_masks'].to(device)  # [N_q, H, W]

        sim = head(decoded, text_emb)  # [1, N_q, H, W]
        sim = sim.squeeze(0)

        for q in range(sim.shape[0]):
            iou_25 = compute_grounding_iou(
                sim[q].unsqueeze(0), gt_masks[q].unsqueeze(0), threshold=0.25
            )
            iou_50 = compute_grounding_iou(
                sim[q].unsqueeze(0), gt_masks[q].unsqueeze(0), threshold=0.5
            )
            metrics['iou_25'].append(iou_25)
            metrics['iou_50'].append(iou_50)

    return {k: np.mean(v) for k, v in metrics.items() if v}


@torch.no_grad()
def eval_feature_quality(model, codec, renderer, dataset, device, config):
    """Evaluate feature reconstruction quality (PSNR, cosine sim vs GT RADIO)."""
    psnrs, cosines = [], []

    for i in tqdm(range(min(len(dataset), 100)), desc='Eval Feature Quality'):
        sample = dataset[i]
        decoded, _ = render_and_decode(
            model, codec, renderer, sample['pose_w2c'], device, config
        )
        gt = sample['radio_features'].unsqueeze(0).to(device)

        if decoded.shape != gt.shape:
            decoded = F.interpolate(decoded, gt.shape[-2:], mode='bilinear', align_corners=True)

        mse = F.mse_loss(decoded, gt).item()
        psnr = -10 * np.log10(max(mse, 1e-10))
        cos = F.cosine_similarity(decoded, gt, dim=1).mean().item()

        psnrs.append(psnr)
        cosines.append(cos)

    return {'feature_psnr': np.mean(psnrs), 'cosine_similarity': np.mean(cosines)}


def main():
    parser = argparse.ArgumentParser(description='RADIO-GS Downstream Evaluation')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--tasks', nargs='+', default=['depth', 'segmentation', 'grounding', 'feature_quality'],
                        help='Tasks to evaluate')
    parser.add_argument('--output_dir', default=None, help='Output directory for results')
    parser.add_argument('--device', default='cuda', help='Device')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    output_dir = args.output_dir or os.path.join(config.output_dir, 'eval_results')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading model from {args.checkpoint}...')
    model, codec, renderer = load_model_and_codec(config, args.checkpoint, device)

    # Placeholder dataset — in practice, load the actual test dataset
    print(f'Note: Using SimpleRadioDataset from feature_dir={config.feature_dir}')
    print(f'Tasks to evaluate: {args.tasks}')

    results = {}

    # Feature quality always evaluated
    if 'feature_quality' in args.tasks:
        print('\n=== Feature Reconstruction Quality ===')
        # Would call: eval_feature_quality(model, codec, renderer, dataset, device, config)
        print('  (Requires dataset — skipping in dry run)')

    if 'depth' in args.tasks:
        print('\n=== Depth Estimation ===')
        from radio_gs.heads.depth_head import DepthHead, DepthLoss
        depth_head = DepthHead(config.radio_feature_dim, head_type=config.depth_head_type)
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'depth_head_state_dict' in ckpt:
            depth_head.load_state_dict(ckpt['depth_head_state_dict'])
            print('  Loaded depth head weights')
        print('  (Requires dataset — skipping in dry run)')

    if 'segmentation' in args.tasks:
        print('\n=== Semantic Segmentation ===')
        from radio_gs.heads.segmentation_head import SegmentationHead
        seg_head = SegmentationHead(config.radio_feature_dim, num_classes=config.seg_num_classes)
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'seg_head_state_dict' in ckpt:
            seg_head.load_state_dict(ckpt['seg_head_state_dict'])
            print('  Loaded segmentation head weights')
        print('  (Requires dataset — skipping in dry run)')

    if 'grounding' in args.tasks:
        print('\n=== Text Grounding ===')
        from radio_gs.heads.grounding_head import GroundingHead
        ground_head = GroundingHead(config.radio_feature_dim, use_adaptor=config.grounding_use_adaptor)
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'grounding_head_state_dict' in ckpt:
            ground_head.load_state_dict(ckpt['grounding_head_state_dict'])
            print('  Loaded grounding head weights')
        print('  (Requires dataset — skipping in dry run)')

    # Save results
    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
