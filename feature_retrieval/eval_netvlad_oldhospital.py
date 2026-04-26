#!/usr/bin/env python3
"""
NetVLAD Retrieval Baseline for OldHospital (Cambridge Landmarks)
================================================================
Extracts NetVLAD 4096-d descriptors from train/test images,
does top-1 cosine retrieval, and evaluates with the same thresholds
as the pose regression experiments.

Usage:
    CUDA_VISIBLE_DEVICES=3 python feature_retrieval/eval_netvlad_oldhospital.py --gpu 0
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_retrieval.retrievers.netvlad_retrieval import NetVLADModel

from PIL import Image
import torchvision.transforms as T


def parse_cambridge_dataset(txt_path):
    """Parse Cambridge Landmarks dataset_{train,test}.txt.
    Returns: list of (image_name, center_xyz, quat_wpqr)
    """
    entries = []
    with open(txt_path) as f:
        lines = f.readlines()
    for line in lines[3:]:  # skip 3 header lines
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        name = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        w, p, q, r = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        entries.append((name, np.array([x, y, z], dtype=np.float32),
                        np.array([w, p, q, r], dtype=np.float32)))
    return entries


def quat_to_rotmat(quat):
    """Convert w-first quaternion to 3x3 rotation matrix (produces R_w2c)."""
    w, x, y, z = quat
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def entries_to_w2c(entries):
    """Convert Cambridge entries to w2c 4x4 matrices.
    Cambridge quaternion = R_w2c, center = camera center in world.
    w2c: t_w2c = -R_w2c @ center
    """
    poses = []
    for name, center, quat in entries:
        R = quat_to_rotmat(quat)  # R_w2c
        t = -R @ center
        T_w2c = np.eye(4, dtype=np.float32)
        T_w2c[:3, :3] = R
        T_w2c[:3, 3] = t
        poses.append(T_w2c)
    return np.stack(poses)


def extract_netvlad_descriptors(model, dataset_dir, entries, resize_max=640, batch_size=8):
    """Extract NetVLAD descriptors for a list of (name, ...) entries."""
    transform = T.ToTensor()  # [0, 1]
    all_descs = []

    for i in range(0, len(entries), batch_size):
        batch_entries = entries[i:i + batch_size]
        images = []
        for name, _, _ in batch_entries:
            img_path = os.path.join(dataset_dir, name)
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            if max(w, h) > resize_max:
                scale = resize_max / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            images.append(transform(img))

        # Pad to same size
        max_h = max(t.shape[1] for t in images)
        max_w = max(t.shape[2] for t in images)
        batch = torch.zeros(len(images), 3, max_h, max_w)
        for j, t in enumerate(images):
            batch[j, :, :t.shape[1], :t.shape[2]] = t

        batch = batch.to(model.device)
        with torch.no_grad():
            descs = model(batch).cpu().numpy()
        all_descs.append(descs)

        if (i // batch_size) % 20 == 0:
            print(f"  [{i+len(batch_entries)}/{len(entries)}]")

    return np.concatenate(all_descs, axis=0).astype(np.float32)


def geodesic_error_deg(R1, R2):
    """Rotation error in degrees between two 3x3 rotation matrices."""
    R_rel = R1 @ R2.T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def evaluate(db_entries, db_poses_w2c, db_descs,
             query_entries, query_poses_w2c, query_descs):
    """Evaluate top-1 retrieval with our standard thresholds."""

    # Build FAISS index
    d = db_descs.shape[1]
    db_norm = db_descs.copy()
    faiss.normalize_L2(db_norm)
    index = faiss.IndexFlatIP(d)
    index.add(db_norm)

    q_norm = query_descs.copy()
    faiss.normalize_L2(q_norm)
    scores, indices = index.search(q_norm, 10)  # top-10

    N = len(query_entries)
    trans_errors = np.zeros(N)
    rot_errors = np.zeros(N)
    retrieved_names = []

    for qi in range(N):
        # GT
        gt_center = query_entries[qi][1]
        gt_R = query_poses_w2c[qi][:3, :3]

        # Retrieved (top-1)
        ri = int(indices[qi, 0])
        ret_center = db_entries[ri][1]
        ret_R = db_poses_w2c[ri][:3, :3]

        trans_errors[qi] = np.linalg.norm(gt_center - ret_center)
        rot_errors[qi] = geodesic_error_deg(gt_R, ret_R)
        retrieved_names.append(db_entries[ri][0])

    # Metrics
    results = {
        'rotation_deg': {
            'median': float(np.median(rot_errors)),
            'mean': float(np.mean(rot_errors)),
            'std': float(np.std(rot_errors)),
            'p25': float(np.percentile(rot_errors, 25)),
            'p75': float(np.percentile(rot_errors, 75)),
            'p90': float(np.percentile(rot_errors, 90)),
            'p95': float(np.percentile(rot_errors, 95)),
        },
        'translation_m': {
            'median': float(np.median(trans_errors)),
            'mean': float(np.mean(trans_errors)),
            'std': float(np.std(trans_errors)),
            'p25': float(np.percentile(trans_errors, 25)),
            'p75': float(np.percentile(trans_errors, 75)),
            'p90': float(np.percentile(trans_errors, 90)),
            'p95': float(np.percentile(trans_errors, 95)),
        },
        'translation_mm': {
            'median': float(np.median(trans_errors) * 1000),
            'mean': float(np.mean(trans_errors) * 1000),
        },
        'recall': {},
        'individual_pass_rates': {},
    }

    # Recall thresholds (same as pose_regressor.py)
    thresholds = [
        ('5deg_1m', 5, 1.0),
        ('10deg_2m', 10, 2.0),
        ('15deg_5m', 15, 5.0),
        ('25deg_5m', 25, 5.0),
    ]
    for name, rot_th, trans_th in thresholds:
        rot_pass = (rot_errors < rot_th)
        trans_pass = (trans_errors < trans_th)
        both_pass = rot_pass & trans_pass
        results['recall'][name] = float(both_pass.mean()) * 100
        results['individual_pass_rates'][name] = {
            'rot_pass': float(rot_pass.mean()) * 100,
            'trans_pass': float(trans_pass.mean()) * 100,
            'both_pass': float(both_pass.mean()) * 100,
        }

    # Also compute top-K spatial oracle
    results['spatial_oracle'] = {}
    for K in [1, 3, 5, 10]:
        best_trans = np.zeros(N)
        best_rot = np.zeros(N)
        for qi in range(N):
            gt_center = query_entries[qi][1]
            gt_R = query_poses_w2c[qi][:3, :3]
            topk_idx = indices[qi, :K]
            dists = np.array([np.linalg.norm(gt_center - db_entries[int(ri)][1])
                              for ri in topk_idx])
            best_i = topk_idx[np.argmin(dists)]
            best_trans[qi] = np.min(dists)
            best_rot[qi] = geodesic_error_deg(gt_R, db_poses_w2c[int(best_i)][:3, :3])

        results['spatial_oracle'][f'top{K}'] = {
            'rot_median': float(np.median(best_rot)),
            'trans_median_mm': float(np.median(best_trans) * 1000),
            'recall_10deg_2m': float(((best_rot < 10) & (best_trans < 2)).mean()) * 100,
        }

    # Print
    print("\n" + "=" * 70)
    print("NETVLAD RETRIEVAL BASELINE — OldHospital")
    print("=" * 70)
    print(f"DB: {len(db_descs)} images,  Query: {len(query_descs)} images")
    print(f"\nTop-1 Retrieval:")
    print(f"  Rotation  median: {results['rotation_deg']['median']:.2f}°  "
          f"mean: {results['rotation_deg']['mean']:.2f}°")
    print(f"  Translation median: {results['translation_mm']['median']:.0f}mm  "
          f"mean: {results['translation_m']['mean']*1000:.0f}mm")

    print(f"\nRecall (joint threshold):")
    for name, rot_th, trans_th in thresholds:
        r = results['individual_pass_rates'][name]
        print(f"  R@{rot_th}°/{trans_th}m: {results['recall'][name]:.1f}%  "
              f"(rot_pass={r['rot_pass']:.1f}%, trans_pass={r['trans_pass']:.1f}%)")

    print(f"\nSpatial Oracle from Top-K:")
    for K in [1, 3, 5, 10]:
        so = results['spatial_oracle'][f'top{K}']
        print(f"  Top-{K:>2}: rot={so['rot_median']:.2f}°  "
              f"trans={so['trans_median_mm']:.0f}mm  "
              f"R@10°/2m={so['recall_10deg_2m']:.1f}%")

    print("=" * 70)

    return results, trans_errors, rot_errors


def main():
    parser = argparse.ArgumentParser(description='NetVLAD Baseline for OldHospital')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--output_dir', type=str,
                        default='output/feature_retrieval/pose_regression/exp07_netvlad_baseline')
    parser.add_argument('--resize_max', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse data
    print("Parsing dataset...")
    train_entries = parse_cambridge_dataset(os.path.join(args.dataset_dir, 'dataset_train.txt'))
    test_entries = parse_cambridge_dataset(os.path.join(args.dataset_dir, 'dataset_test.txt'))
    print(f"  Train: {len(train_entries)}, Test: {len(test_entries)}")

    # Convert to w2c
    train_w2c = entries_to_w2c(train_entries)
    test_w2c = entries_to_w2c(test_entries)

    # Load NetVLAD model
    print("Loading NetVLAD model...")
    model = NetVLADModel(device=device)

    # Check for cached descriptors
    db_cache = out_dir / 'train_descs.npy'
    q_cache = out_dir / 'test_descs.npy'

    if db_cache.exists():
        print("Loading cached train descriptors...")
        train_descs = np.load(db_cache)
    else:
        print("Extracting train descriptors...")
        t0 = time.time()
        train_descs = extract_netvlad_descriptors(
            model, args.dataset_dir, train_entries,
            resize_max=args.resize_max, batch_size=args.batch_size)
        print(f"  Done: {train_descs.shape} in {time.time()-t0:.1f}s")
        np.save(db_cache, train_descs)

    if q_cache.exists():
        print("Loading cached test descriptors...")
        test_descs = np.load(q_cache)
    else:
        print("Extracting test descriptors...")
        t0 = time.time()
        test_descs = extract_netvlad_descriptors(
            model, args.dataset_dir, test_entries,
            resize_max=args.resize_max, batch_size=args.batch_size)
        print(f"  Done: {test_descs.shape} in {time.time()-t0:.1f}s")
        np.save(q_cache, test_descs)

    # Evaluate
    results, trans_errors, rot_errors = evaluate(
        train_entries, train_w2c, train_descs,
        test_entries, test_w2c, test_descs)

    # Save
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == '__main__':
    main()
