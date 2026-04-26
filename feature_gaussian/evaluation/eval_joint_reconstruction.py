#!/usr/bin/env python3
"""Evaluate multiscale joint 2DGS feature reconstruction and save panels."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from feature_gaussian.legacy_3dgs.train_2dgs_joint import (
    DA3FeatureCache,
    GaussianModel2DGSJoint,
    build_da3_image_order,
    render_rgb_2dgs,
    render_features_2dgs,
)
from feature_gaussian.legacy_3dgs.train_2dgs_joint_v2 import load_scene_colmap
from feature_gaussian.legacy_3dgs.train_2dgs_joint_v3 import load_config, visualize_comparison_v3
from feature_field.utils.region_metrics import compute_region_masks, score_regions, summarize_region_scores


def _ensure_bchw_alpha(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.ndim == 4 and alpha.shape[-1] == 1:
        return alpha.permute(0, 3, 1, 2)
    if alpha.ndim == 3:
        return alpha.unsqueeze(1)
    return alpha


def _ensure_bchw_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 4 and depth.shape[-1] == 1:
        return depth.permute(0, 3, 1, 2)
    if depth.ndim == 3:
        return depth.unsqueeze(1)
    return depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate joint 2DGS multiscale reconstruction")
    parser.add_argument("--config", required=True, help="Training YAML config")
    parser.add_argument("--ply_path", default=None, help="Optional override for point_cloud.ply")
    parser.add_argument("--features_dir", default=None, help="Optional override for saved feature embeddings")
    parser.add_argument("--output_dir", default=None, help="Optional output directory")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--sample_indices", type=int, nargs="*", default=None)
    parser.add_argument("--camera_split", choices=["train", "test", "all", "auto"], default=None)
    parser.add_argument("--max_metrics_cameras", type=int, default=None)
    return parser.parse_args()


def _summarize_metrics(metrics: list[dict]) -> dict:
    summary: dict[str, dict] = {
        'num_cameras': len(metrics),
        'scales': {},
    }
    if not metrics:
        return summary

    scale_names = metrics[0]['scales'].keys()
    for scale in scale_names:
        cos_vals = []
        l1_vals = []
        region_accum: dict[str, list[float]] = {}
        for entry in metrics:
            scale_entry = entry['scales'][scale]
            cos_vals.append(scale_entry['cosine_mean'])
            l1_vals.append(scale_entry['l1_mean'])
            for region_name, region_stats in scale_entry['region_cosine_mean'].items():
                if region_stats['count'] <= 0:
                    continue
                region_accum.setdefault(region_name, []).append(region_stats['mean'])
        summary['scales'][scale] = {
            'cosine_mean': float(sum(cos_vals) / len(cos_vals)),
            'l1_mean': float(sum(l1_vals) / len(l1_vals)),
            'region_cosine_mean': {
                region_name: {
                    'mean': float(sum(values) / len(values)),
                    'count': len(values),
                }
                for region_name, values in region_accum.items()
            },
        }
    return summary


def _choose_cameras(cams: list, num_samples: int, sample_indices: list[int] | None) -> list:
    if sample_indices:
        chosen = [cams[idx] for idx in sample_indices if 0 <= idx < len(cams)]
        if chosen:
            return chosen
    if len(cams) <= num_samples:
        return cams
    indices = torch.linspace(0, len(cams) - 1, steps=num_samples).round().to(dtype=torch.long).tolist()
    return [cams[idx] for idx in indices]


def _default_ply_path(cfg: dict) -> Path:
    base_dir = Path(cfg['output_dir']) / cfg['exp_name'] / 'point_cloud'
    for candidate in [base_dir / 'best' / 'point_cloud.ply', base_dir / f"iteration_{cfg['training']['iterations']}" / 'point_cloud.ply']:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find saved point cloud under {base_dir}")


def _default_features_dir(cfg: dict) -> Path:
    base_dir = Path(cfg['output_dir']) / cfg['exp_name']
    for candidate in [base_dir / 'features_best', base_dir / 'features']:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find saved features under {base_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg['output_dir']) / cfg['exp_name'] / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    dcfg = cfg['dataset']
    scales = dcfg['feature_scales']
    if isinstance(scales, str):
        scales = [item.strip() for item in scales.split(',') if item.strip()]

    train_cams, test_cams, _, _, cameras_extent = load_scene_colmap(dcfg['source_dir'], dcfg.get('images', ''))
    camera_split = args.camera_split or args.split
    if camera_split == 'train':
        selected_pool = train_cams
    elif camera_split == 'test':
        selected_pool = test_cams
    elif camera_split == 'all':
        selected_pool = [*train_cams, *test_cams]
    else:
        selected_pool = test_cams if test_cams else train_cams

    feat_cache = DA3FeatureCache(dcfg['feature_dir'], scales)
    images_subdir = dcfg.get('images', '')
    if images_subdir:
        images_dir = Path(dcfg['source_dir']) / images_subdir
    else:
        source_dir = Path(dcfg['source_dir'])
        if any(source_dir.glob('seq*')):
            images_dir = source_dir
        elif (source_dir / 'images').is_dir():
            images_dir = source_dir / 'images'
        else:
            images_dir = source_dir

    da3_name_to_fid = build_da3_image_order(str(images_dir))
    available_fids = feat_cache.frame_ids(scales[0])
    cam_to_fid = {}
    for cam in [*train_cams, *test_cams]:
        fid = da3_name_to_fid.get(cam.image_name)
        if fid is not None and fid in available_fids:
            cam_to_fid[cam.uid] = fid

    valid_cams = [cam for cam in selected_pool if cam.uid in cam_to_fid]
    if not valid_cams:
        raise RuntimeError(f"No {args.split} cameras matched cached features")
    chosen_cams = _choose_cameras(valid_cams, args.num_samples, args.sample_indices)
    metric_cams = valid_cams[:args.max_metrics_cameras] if args.max_metrics_cameras is not None else valid_cams

    cam0 = train_cams[0]
    tanfovx = math.tan(cam0.FovX * 0.5)
    tanfovy = math.tan(cam0.FovY * 0.5)
    img_fx = cam0.width / (2 * tanfovx)
    img_fy = cam0.height / (2 * tanfovy)
    scale_infos = {}
    for scale in scales:
        dim, height, width = feat_cache.info(scale)
        sx, sy = width / cam0.width, height / cam0.height
        scale_infos[scale] = {
            'dim': dim,
            'h': height,
            'w': width,
            'K': torch.tensor(
                [
                    [img_fx * sx, 0, cam0.width * sx / 2.0],
                    [0, img_fy * sy, cam0.height * sy / 2.0],
                    [0, 0, 1],
                ],
                device='cuda',
                dtype=torch.float32,
            ),
        }

    feature_dims = {scale: scale_infos[scale]['dim'] for scale in scales}
    gaussians = GaussianModel2DGSJoint(
        sh_degree=cfg['model']['sh_degree'],
        feature_scales=feature_dims,
    )
    ply_path = Path(args.ply_path) if args.ply_path else _default_ply_path(cfg)
    gaussians.load_ply(str(ply_path))
    gaussians.spatial_lr_scale = cameras_extent
    features_dir = Path(args.features_dir) if args.features_dir else _default_features_dir(cfg)
    gaussians.load_features(features_dir)

    vis_cfg = copy.deepcopy(cfg)
    vis_cfg['training']['vis_frames'] = list(range(len(chosen_cams)))
    visualize_comparison_v3(
        gaussians,
        chosen_cams,
        cam_to_fid,
        feat_cache,
        scale_infos,
        scales,
        vis_cfg,
        iteration=0,
        output_dir=str(output_dir),
        appearance_net=None,
        masks=None,
    )

    metrics = []
    bg_color = torch.tensor(
        [1, 1, 1] if cfg['model']['white_background'] else [0, 0, 0],
        dtype=torch.float32,
        device='cuda',
    )
    longest_edge = int(cfg['training'].get('longest_edge', 0))
    for cam in metric_cams:
        viewmat = cam.get_world_view_transform()
        fid = cam_to_fid[cam.uid]
        render_pkg = render_rgb_2dgs(gaussians, cam, bg_color, longest_edge=longest_edge)
        alpha = _ensure_bchw_alpha(render_pkg['rend_alpha'])
        depth = _ensure_bchw_depth(render_pkg['depth'])
        entry = {
            'image_name': cam.image_name,
            'frame_id': int(fid),
            'scales': {},
        }
        for scale in scales:
            gt_feat = feat_cache.get(scale, fid)
            info = scale_infos[scale]
            rendered = render_features_2dgs(
                gaussians,
                viewmat,
                gaussians.get_feature(scale),
                info['h'],
                info['w'],
                info['K'],
            )
            pred = rendered.unsqueeze(0)
            target = gt_feat.unsqueeze(0)
            alpha_scale = alpha
            depth_scale = depth
            if alpha_scale.shape[-2:] != (info['h'], info['w']):
                alpha_scale = F.interpolate(alpha_scale, (info['h'], info['w']), mode='bilinear', align_corners=False)
            if depth_scale.shape[-2:] != (info['h'], info['w']):
                depth_scale = F.interpolate(depth_scale, (info['h'], info['w']), mode='bilinear', align_corners=False)
            region_masks, _ = compute_region_masks(depth=depth_scale, alpha=alpha_scale)
            cosine_map = F.cosine_similarity(pred, target, dim=1)
            valid = region_masks['foreground'].squeeze(1)
            cos_mean = cosine_map[valid].mean().item() if valid.any() else float('nan')
            l1_mean = F.l1_loss(rendered, gt_feat).item()
            entry['scales'][scale] = {
                'cosine_mean': float(cos_mean),
                'l1_mean': float(l1_mean),
                'region_cosine_mean': summarize_region_scores(score_regions(cosine_map, region_masks)),
            }
        metrics.append(entry)

    metrics_path = output_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(_summarize_metrics(metrics), indent=2) + '\n', encoding='utf-8')
    print(f"Saved joint 2DGS evaluation to {output_dir}")


if __name__ == '__main__':
    main()
