#!/usr/bin/env python3
"""Lightweight deterministic DCFF evaluation without writing images."""

import argparse
import json
import os
import numpy as np
import torch
from PIL import Image

from feature_field.visualize_feature_comparison import (
    _build_dcff_eval_components,
    _apply_dcff_postprocess,
    cosine_similarity_map,
    _cam_to_viewmat,
    _cam_to_K,
)
from feature_field.dcff.radio_teacher import CachedFeatureTeacher
from feature_field.utils.region_metrics import (
    compute_region_masks,
    empty_region_score_dict,
    extend_region_scores,
    score_regions,
    summarize_region_scores,
)
from feature_field.utils.scene_colmap import load_scene_colmap, build_da3_image_order


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DCFF feature metrics without visualization output")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_dir", default="dataset/OldHospital")
    parser.add_argument("--feature_dir", default="feature_extract/output/features_radio_dual/OldHospital_pilot")
    parser.add_argument("--images_subdir", default="", help="Optional image subdir, e.g. processed")
    parser.add_argument("--camera_split", default="test", choices=["train", "test", "all", "auto"])
    parser.add_argument("--camera_indices", default=None, help="Comma-separated indices into the selected split")
    parser.add_argument("--max_cameras", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _render_size(cam, longest_edge):
    width, height = Image.open(cam.image).size
    scale = longest_edge / max(width, height)
    if scale < 1.0:
        width = int(width * scale)
        height = int(height * scale)
    return height, width


def _mean_from_map(sim_map, mask):
    scores = []
    valid = mask.squeeze(1) > 0.5
    for b in range(sim_map.shape[0]):
        vb = valid[b]
        if vb.any():
            scores.append(float(sim_map[b][vb].mean().item()))
        else:
            scores.append(float('nan'))
    return scores


def main():
    args = parse_args()
    device = torch.device(args.device)

    bundle = _build_dcff_eval_components(args.checkpoint, device)
    gaussians = bundle['gaussians']
    renderer = bundle['renderer']
    feat_sharp_fine = bundle['feat_sharp_fine']
    feat_select = bundle['feat_select']
    longest_edge = bundle['longest_edge']
    coarse_downsample = bundle['coarse_downsample']

    train_cams, test_cams, _, _, _ = load_scene_colmap(args.source_dir, args.images_subdir)
    radio_cache = CachedFeatureTeacher(args.feature_dir)
    feat_h, feat_w = radio_cache.feat_h, radio_cache.feat_w

    if args.images_subdir:
        images_dir = os.path.join(args.source_dir, args.images_subdir)
    else:
        images_dir = (
            os.path.join(args.source_dir, 'images')
            if os.path.isdir(os.path.join(args.source_dir, 'images'))
            else args.source_dir
        )
    da3_name_to_fid = build_da3_image_order(images_dir)

    cam_to_fid = {}
    for cam in train_cams + test_cams:
        fid = da3_name_to_fid.get(cam.image_name)
        if fid is not None and fid in radio_cache.frame_ids:
            cam_to_fid[cam.uid] = fid

    if args.camera_split == 'train':
        cam_pool = train_cams
    elif args.camera_split == 'test':
        cam_pool = test_cams
    elif args.camera_split == 'all':
        cam_pool = train_cams + test_cams
    else:
        cam_pool = test_cams if test_cams else train_cams

    valid_cams = [c for c in cam_pool if c.uid in cam_to_fid]
    if args.camera_indices:
        indices = [int(x) for x in args.camera_indices.split(',') if x.strip()]
        valid_cams = [valid_cams[i] for i in indices if 0 <= i < len(valid_cams)]
    elif args.max_cameras is not None:
        valid_cams = valid_cams[:args.max_cameras]

    grouped_cams = {}
    for cam in valid_cams:
        grouped_cams.setdefault(_render_size(cam, longest_edge), []).append(cam)

    fine_scores = []
    coarse_scores = []
    fine_region_scores = empty_region_score_dict()
    coarse_region_scores = empty_region_score_dict()
    per_camera = []

    with torch.no_grad():
        processed = 0
        coarse_h, coarse_w = feat_h // 2, feat_w // 2
        for (h_render, w_render), cams_in_group in grouped_cams.items():
            for start in range(0, len(cams_in_group), args.batch_size):
                cams = cams_in_group[start:start + args.batch_size]
                fids = [cam_to_fid[c.uid] for c in cams]
                geo_batch, sem_batch = [], []
                for fid in fids:
                    geo, sem = radio_cache.get(fid)
                    geo_batch.append(geo)
                    sem_batch.append(sem)
                geo_target = torch.stack(geo_batch, dim=0).to(device)
                sem_target = torch.stack(sem_batch, dim=0).to(device)

                viewmat = torch.cat([_cam_to_viewmat(cam, device) for cam in cams], dim=0)
                K = torch.cat([_cam_to_K(cam, w_render, h_render, device) for cam in cams], dim=0)

                result = renderer(
                    gaussians,
                    viewmat=viewmat,
                    K=K,
                    width=w_render,
                    height=h_render,
                    render_coarse=True,
                    feature_height=feat_h,
                    feature_width=feat_w,
                )
                result = _apply_dcff_postprocess(
                    result,
                    feat_h,
                    feat_w,
                    feat_sharp_fine=feat_sharp_fine,
                    feat_select=feat_select,
                )

                fine_pred = result['fine_features']
                if fine_pred.shape[-2:] != (feat_h, feat_w):
                    fine_pred = torch.nn.functional.interpolate(fine_pred, (feat_h, feat_w), mode='bilinear', align_corners=False)
                alpha_fine = result['alpha']
                if alpha_fine.shape[-2:] != (feat_h, feat_w):
                    alpha_fine = torch.nn.functional.interpolate(alpha_fine, (feat_h, feat_w), mode='bilinear', align_corners=False)
                mask_fine = (alpha_fine > 0.5).float()
                fine_cos_map = cosine_similarity_map(fine_pred, geo_target, mask_fine)
                fine_batch_scores = _mean_from_map(fine_cos_map, mask_fine)
                fine_regions, _ = compute_region_masks(depth=result['depth'], alpha=alpha_fine)
                fine_region_batch = score_regions(fine_cos_map, fine_regions)
                extend_region_scores(fine_region_scores, fine_region_batch)

                coarse_target = sem_target
                if coarse_downsample:
                    coarse_target = torch.nn.functional.interpolate(sem_target, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
                coarse_pred = result['coarse_features']
                if coarse_pred.shape[-2:] != coarse_target.shape[-2:]:
                    coarse_pred = torch.nn.functional.interpolate(coarse_pred, coarse_target.shape[-2:], mode='bilinear', align_corners=False)
                alpha_coarse = result['alpha']
                if alpha_coarse.shape[-2:] != coarse_target.shape[-2:]:
                    alpha_coarse = torch.nn.functional.interpolate(alpha_coarse, coarse_target.shape[-2:], mode='bilinear', align_corners=False)
                mask_coarse = (alpha_coarse > 0.5).float()
                coarse_cos_map = cosine_similarity_map(coarse_pred, coarse_target, mask_coarse)
                coarse_batch_scores = _mean_from_map(coarse_cos_map, mask_coarse)
                coarse_depth = result['depth']
                if coarse_depth.shape[-2:] != coarse_target.shape[-2:]:
                    coarse_depth = torch.nn.functional.interpolate(coarse_depth, coarse_target.shape[-2:], mode='bilinear', align_corners=False)
                coarse_regions, _ = compute_region_masks(depth=coarse_depth, alpha=alpha_coarse)
                coarse_region_batch = score_regions(coarse_cos_map, coarse_regions)
                extend_region_scores(coarse_region_scores, coarse_region_batch)

                for cam_idx, (cam, fid, fine, coarse) in enumerate(zip(cams, fids, fine_batch_scores, coarse_batch_scores)):
                    processed += 1
                    fine_scores.append(fine)
                    coarse_scores.append(coarse)
                    per_camera.append({
                        'index': processed - 1,
                        'uid': int(cam.uid),
                        'fid': int(fid),
                        'image_name': cam.image_name,
                        'fine_cos': fine,
                        'coarse_cos': coarse,
                        'fine_regions': {name: fine_region_batch[name][cam_idx] for name in fine_region_batch},
                        'coarse_regions': {name: coarse_region_batch[name][cam_idx] for name in coarse_region_batch},
                    })
                    print(f"[{processed}/{len(valid_cams)}] {cam.image_name}: fine={fine:.4f} coarse={coarse:.4f}")

    result = {
        'checkpoint': args.checkpoint,
        'split': args.camera_split,
        'num_cameras': len(per_camera),
        'fine_cos': float(np.mean(fine_scores)) if fine_scores else 0.0,
        'fine_cos_std': float(np.std(fine_scores)) if fine_scores else 0.0,
        'fine_region_cos': summarize_region_scores(fine_region_scores),
        'coarse_cos': float(np.mean(coarse_scores)) if coarse_scores else 0.0,
        'coarse_cos_std': float(np.std(coarse_scores)) if coarse_scores else 0.0,
        'coarse_region_cos': summarize_region_scores(coarse_region_scores),
        'per_camera': per_camera,
    }
    print(json.dumps({k: v for k, v in result.items() if k != 'per_camera'}, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
