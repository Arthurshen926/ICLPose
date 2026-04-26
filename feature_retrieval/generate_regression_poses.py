#!/usr/bin/env python3
"""
Generate regression-predicted poses for pipeline integration.
=============================================================

Loads the exp01 PoseRegressorMLP model, runs inference on all test images,
converts c2w predictions to w2c format, and saves as .npz for use by
evaluate_pipeline.py --init_method regression|regression_loftr.

Output .npz keys:
    img_names:        (N,) array of image name strings
    poses_w2c:        (N, 4, 4) float32 w2c pose matrices
    camera_centers:   (N, 3) float64 predicted camera centers (in world coords)
    rot_errors_deg:   (N,) float64 per-image rotation error vs GT (for analysis)
    trans_errors_m:   (N,) float64 per-image translation error vs GT (for analysis)

Usage:
    python feature_retrieval/generate_regression_poses.py \
        --checkpoint output/feature_retrieval/pose_regression/exp01_direct_mlp/model_best.pt \
        --output output/feature_retrieval/pose_regression/exp01_direct_mlp/test_poses_w2c.npz \
        --gpu 0
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_retrieval.pose_regressor import (
    PoseRegressorMLP,
    PoseRegressionDataset,
    quaternion_to_matrix,
)


def c2w_to_w2c(R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    """Convert regression predictions to 4x4 w2c matrix.

    IMPORTANT: Despite the Cambridge Landmarks convention labelling quaternions
    as "c2w", the stored quaternions actually produce w2c rotation matrices
    (identical to COLMAP). The regression model was trained against these,
    so its rotation output is already R_w2c.

    Args:
        R_c2w: (3, 3) rotation matrix — actually R_w2c (see above)
        t_c2w: (3,) camera center in world coordinates

    Returns:
        (4, 4) float32 w2c matrix
    """
    w2c = np.eye(4, dtype=np.float32)
    R_w2c = R_c2w  # Model output IS w2c, not c2w — see docstring
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = -R_w2c @ t_c2w
    return w2c


def load_gt_w2c_poses(dataset_dir: str, split: str = "test"):
    """Load GT w2c poses from Cambridge Landmarks dataset files.

    Returns dict: img_name -> 4x4 w2c numpy array
    """
    from data.radio_loc_dataset import read_colmap_images
    colmap_dir = os.path.join(dataset_dir, "sparse", "0")
    images = read_colmap_images(os.path.join(colmap_dir, "images.bin"))

    split_file = os.path.join(dataset_dir, f"dataset_{split}.txt")
    with open(split_file) as f:
        lines = f.readlines()

    split_names = set()
    for line in lines[3:]:
        parts = line.strip().split()
        if parts:
            split_names.add(parts[0])

    gt_poses = {}
    for img_id, meta in sorted(images.items()):
        if meta.name not in split_names:
            continue
        # COLMAP stores w2c directly
        from pose_refine.evaluate_pipeline import _qvec_to_rotmat
        R = _qvec_to_rotmat(meta.qvec)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = meta.tvec
        gt_poses[meta.name] = w2c

    return gt_poses


def main():
    parser = argparse.ArgumentParser(description="Generate regression init poses for pipeline")
    parser.add_argument("--checkpoint", type=str,
                        default="output/feature_retrieval/pose_regression/exp01_direct_mlp/model_best.pt")
    parser.add_argument("--summary_matrix", type=str,
                        default="output/feature_extract/features_radio_dual/OldHospital_pilot/summary_matrix.pt")
    parser.add_argument("--dataset_dir", type=str,
                        default="dataset/OldHospital")
    parser.add_argument("--output", type=str,
                        default="output/feature_retrieval/pose_regression/exp01_direct_mlp/test_poses_w2c.npz")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    trans_mean = ckpt["norm_params"]["mean"].to(device)
    trans_std = ckpt["norm_params"]["std"].to(device)

    # Build model
    model = PoseRegressorMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Load test data
    test_data = PoseRegressionDataset(
        args.summary_matrix, args.dataset_dir, split="test", device=device
    )
    print(f"Test samples: {test_data.N}")

    # Also load GT w2c for error computation
    gt_w2c_dict = load_gt_w2c_poses(args.dataset_dir, "test")
    print(f"GT poses loaded: {len(gt_w2c_dict)}")

    # Run inference
    img_names = []
    poses_w2c = []
    camera_centers = []
    rot_errors = []
    trans_errors = []

    with torch.no_grad():
        # Process all at once (small dataset)
        features = test_data.features  # (N, 2560)
        trans_norm, rot_mat = model(features)

        # Denormalize translation -> c2w position (world coords)
        trans_world = trans_norm * trans_std + trans_mean  # (N, 3)
        rot_c2w = rot_mat  # (N, 3, 3) — model predicts c2w rotation

        trans_np = trans_world.cpu().numpy()  # (N, 3) float32
        rot_np = rot_c2w.cpu().numpy()        # (N, 3, 3) float32

    for i in range(test_data.N):
        name = test_data.names[i]
        R_c2w = rot_np[i]    # (3, 3)
        t_c2w = trans_np[i]   # (3,) = camera center in world

        # Convert to w2c
        w2c = c2w_to_w2c(R_c2w, t_c2w)

        img_names.append(name)
        poses_w2c.append(w2c)
        camera_centers.append(t_c2w.astype(np.float64))

        # Compute error vs GT
        if name in gt_w2c_dict:
            gt_w2c = gt_w2c_dict[name]
            # Rotation error
            R_rel = w2c[:3, :3].T @ gt_w2c[:3, :3].astype(np.float32)
            trace = np.clip(np.trace(R_rel), -1.0, 3.0)
            cos_a = np.clip((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
            rot_err = float(np.degrees(np.arccos(cos_a)))
            # Translation error (camera center distance)
            gt_center = -gt_w2c[:3, :3].T @ gt_w2c[:3, 3]
            trans_err = float(np.linalg.norm(t_c2w - gt_center))
        else:
            rot_err = -1.0
            trans_err = -1.0

        rot_errors.append(rot_err)
        trans_errors.append(trans_err)

    poses_w2c = np.array(poses_w2c, dtype=np.float32)         # (N, 4, 4)
    camera_centers = np.array(camera_centers, dtype=np.float64)  # (N, 3)
    rot_errors = np.array(rot_errors, dtype=np.float64)
    trans_errors = np.array(trans_errors, dtype=np.float64)

    # Summary stats
    valid = rot_errors >= 0
    print(f"\nRegression pose quality ({valid.sum()} test images):")
    print(f"  Rotation:    median {np.median(rot_errors[valid]):.2f}°, "
          f"mean {np.mean(rot_errors[valid]):.2f}°")
    print(f"  Translation: median {np.median(trans_errors[valid])*1000:.0f}mm, "
          f"mean {np.mean(trans_errors[valid])*1000:.0f}mm")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        img_names=np.array(img_names),
        poses_w2c=poses_w2c,
        camera_centers=camera_centers,
        rot_errors_deg=rot_errors,
        trans_errors_m=trans_errors,
    )
    print(f"\nSaved {len(img_names)} poses to {args.output}")


if __name__ == "__main__":
    main()
