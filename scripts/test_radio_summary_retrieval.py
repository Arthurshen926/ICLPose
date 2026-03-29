#!/usr/bin/env python3
"""
测试 RADIO Global Summary 向量的定位初始化能力

使用 RADIO 的 2560-d summary 向量进行图像检索:
  - 构建训练集的 summary 矩阵 (N_train, 2560)
  - 对测试集每张图像，找 top-K 最近邻
  - 最近邻的位姿作为初始位姿估计
  - 评估 top-1/5/10 位姿精度

用法:
    CUDA_VISIBLE_DEVICES=5 python scripts/test_radio_summary_retrieval.py \
        --feature_dir output/features_radio/OldHospital_indexed \
        --source_dir dataset/OldHospital \
        --top_k 5
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_poses_cambridge(source_dir):
    """Load poses from Cambridge-style dataset.
    Returns dict[image_name] -> 4x4 c2w matrix."""
    poses = {}
    for split_file in ['dataset_train.txt', 'dataset_test.txt']:
        fpath = os.path.join(source_dir, split_file)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            lines = f.readlines()
        for line in lines[3:]:  # Skip header
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            name = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            w, p, q, r = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            # Quaternion to rotation matrix (w, x, y, z convention)
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat([p, q, r, w]).as_matrix()
            c2w = np.eye(4)
            c2w[:3, :3] = rot
            c2w[:3, 3] = [x, y, z]
            poses[name] = c2w
    return poses


def build_image_order(source_dir):
    """Build image_name -> frame_id mapping (same as DA3FeatureCache)."""
    import glob
    images = []
    seq_dirs = sorted(glob.glob(os.path.join(source_dir, "seq*")))
    if seq_dirs and all(os.path.isdir(d) for d in seq_dirs):
        global_idx = 0
        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            for fname in sorted(os.listdir(seq_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    images.append((global_idx, f"{seq_name}/{fname}"))
                    global_idx += 1
    return {name: idx for idx, name in images}


def rotation_error_deg(R1, R2):
    """Compute rotation error in degrees."""
    R_diff = R1[:3, :3] @ R2[:3, :3].T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    return np.degrees(angle)


def translation_error(t1, t2):
    """Compute translation error in meters."""
    return np.linalg.norm(t1 - t2)


def main():
    parser = argparse.ArgumentParser(description='Test RADIO summary retrieval')
    parser.add_argument('--feature_dir', type=str, required=True,
                        help='RADIO feature directory with summary_matrix.pt')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Dataset directory (e.g., dataset/OldHospital)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of nearest neighbors to evaluate')
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    source_dir = args.source_dir

    # 1. Load summary matrix
    summary_path = feature_dir / 'summary_matrix.pt'
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run extraction with --save_summary first.")
        return
    summary_matrix = torch.load(summary_path, map_location='cpu').float()  # (N, 2560)
    print(f"Summary matrix: {summary_matrix.shape}")

    # 2. Load poses
    poses = load_poses_cambridge(source_dir)
    print(f"Loaded {len(poses)} poses")

    # 3. Build image order
    name_to_fid = build_image_order(source_dir)
    fid_to_name = {v: k for k, v in name_to_fid.items()}
    print(f"Image order: {len(name_to_fid)} images")

    # 4. Split train/test
    train_names = set()
    test_names = set()
    train_path = os.path.join(source_dir, 'dataset_train.txt')
    test_path = os.path.join(source_dir, 'dataset_test.txt')

    for fpath, name_set in [(train_path, train_names), (test_path, test_names)]:
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f.readlines()[3:]:
                    parts = line.strip().split()
                    if len(parts) == 8:
                        name_set.add(parts[0])

    print(f"Train: {len(train_names)}, Test: {len(test_names)}")

    # 5. Build train summary database
    train_fids = []
    train_names_list = []
    for name in sorted(train_names):
        fid = name_to_fid.get(name)
        if fid is not None and fid < summary_matrix.shape[0]:
            train_fids.append(fid)
            train_names_list.append(name)

    train_summaries = summary_matrix[train_fids]  # (N_train, 2560)
    # L2 normalize for cosine similarity
    train_summaries_norm = F.normalize(train_summaries, dim=1)
    print(f"Train database: {train_summaries.shape}")

    # 6. Query each test image
    test_fids = []
    test_names_list = []
    for name in sorted(test_names):
        fid = name_to_fid.get(name)
        if fid is not None and fid < summary_matrix.shape[0]:
            test_fids.append(fid)
            test_names_list.append(name)

    test_summaries = summary_matrix[test_fids]  # (N_test, 2560)
    test_summaries_norm = F.normalize(test_summaries, dim=1)
    print(f"Test queries: {test_summaries.shape}")

    # 7. Compute similarities and find top-K
    # Cosine similarity: (N_test, N_train)
    sim_matrix = test_summaries_norm @ train_summaries_norm.T
    topk_vals, topk_ids = sim_matrix.topk(args.top_k, dim=1)

    # 8. Evaluate pose errors
    rot_errors_topk = {k: [] for k in range(1, args.top_k + 1)}
    trans_errors_topk = {k: [] for k in range(1, args.top_k + 1)}

    for i, test_name in enumerate(test_names_list):
        if test_name not in poses:
            continue
        gt_pose = poses[test_name]

        for k in range(args.top_k):
            nn_idx = topk_ids[i, k].item()
            nn_name = train_names_list[nn_idx]
            if nn_name not in poses:
                continue
            nn_pose = poses[nn_name]

            rot_err = rotation_error_deg(gt_pose, nn_pose)
            trans_err = translation_error(gt_pose[:3, 3], nn_pose[:3, 3])

            # Top-K means best of first K neighbors
            for kk in range(k, args.top_k):
                rot_errors_topk[kk + 1].append(rot_err)
                trans_errors_topk[kk + 1].append(trans_err)
            break  # Only use top-1 for each K threshold

    # Fix: For top-K, we should take the best of K neighbors
    print(f"\n{'='*60}")
    print(f"RADIO Summary Retrieval Results ({len(test_names_list)} test images)")
    print(f"{'='*60}")

    # Re-evaluate properly: for each test image, top-K means best error among K nearest
    for k in [1, 3, 5, min(10, args.top_k)]:
        if k > args.top_k:
            continue
        rot_errs = []
        trans_errs = []
        for i, test_name in enumerate(test_names_list):
            if test_name not in poses:
                continue
            gt_pose = poses[test_name]

            best_rot = float('inf')
            best_trans = float('inf')
            for j in range(k):
                nn_idx = topk_ids[i, j].item()
                nn_name = train_names_list[nn_idx]
                if nn_name not in poses:
                    continue
                nn_pose = poses[nn_name]
                rot_err = rotation_error_deg(gt_pose, nn_pose)
                trans_err = translation_error(gt_pose[:3, 3], nn_pose[:3, 3])
                if trans_err < best_trans:
                    best_rot = rot_err
                    best_trans = trans_err
            rot_errs.append(best_rot)
            trans_errs.append(best_trans)

        rot_errs = np.array(rot_errs)
        trans_errs = np.array(trans_errs)

        print(f"\nTop-{k} Nearest Neighbor (best of {k}):")
        print(f"  Rotation:    median={np.median(rot_errs):.2f}° mean={np.mean(rot_errs):.2f}°")
        print(f"  Translation: median={np.median(trans_errs):.3f}m mean={np.mean(trans_errs):.3f}m")

        # Percentage within thresholds
        for t_thresh, r_thresh in [(0.25, 2), (0.5, 5), (1.0, 10), (5.0, 25)]:
            pct = np.mean((trans_errs < t_thresh) & (rot_errs < r_thresh)) * 100
            print(f"  Within {t_thresh}m/{r_thresh}°: {pct:.1f}%")

    # 9. Show some examples
    print(f"\n{'='*60}")
    print(f"Sample retrievals (first 5 test images):")
    print(f"{'='*60}")
    for i in range(min(5, len(test_names_list))):
        test_name = test_names_list[i]
        if test_name not in poses:
            continue
        gt_pose = poses[test_name]
        print(f"\n  Query: {test_name}")
        print(f"  GT pos: ({gt_pose[0,3]:.2f}, {gt_pose[1,3]:.2f}, {gt_pose[2,3]:.2f})")
        for j in range(min(3, args.top_k)):
            nn_idx = topk_ids[i, j].item()
            nn_name = train_names_list[nn_idx]
            sim = topk_vals[i, j].item()
            if nn_name in poses:
                nn_pose = poses[nn_name]
                rot_err = rotation_error_deg(gt_pose, nn_pose)
                trans_err = translation_error(gt_pose[:3, 3], nn_pose[:3, 3])
                print(f"    #{j+1} {nn_name} (sim={sim:.4f}) "
                      f"err: {trans_err:.3f}m / {rot_err:.2f}°")


if __name__ == '__main__':
    main()
