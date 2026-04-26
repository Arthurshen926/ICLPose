#!/usr/bin/env python3
"""
Prepare stairs scene for feature embedding pipeline.
1. Collect images from seq-02 into a flat directory with rgb_{N}.png naming
2. Generate trajectory file from individual pose files
"""
import os, sys, shutil, glob
import numpy as np
from pathlib import Path

def main():
    dataset_dir = Path("dataset/stairs")
    seq = "seq-02"  # Training split sequence
    seq_dir = dataset_dir / seq
    
    output_base = Path("output/features_multiscale/stairs_seq02")
    rgb_dir = output_base / "rgb_flat"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect color images and poses
    frames = sorted(glob.glob(str(seq_dir / "frame-*.color.png")))
    print(f"Found {len(frames)} frames in {seq}")
    
    poses_c2w = []
    symlink_count = 0
    
    for i, frame_path in enumerate(frames):
        frame_path = Path(frame_path)
        # Parse frame number
        stem = frame_path.stem  # frame-000123.color
        
        # Create symlink: rgb_{i}.png -> original
        link_name = rgb_dir / f"rgb_{i}.png"
        if not link_name.exists():
            os.symlink(frame_path.resolve(), link_name)
        symlink_count += 1
        
        # Load pose
        pose_path = frame_path.parent / frame_path.name.replace('.color.png', '.pose.txt')
        if not pose_path.exists():
            print(f"WARNING: Missing pose for frame {i}: {pose_path}")
            poses_c2w.append(np.eye(4))
            continue
        pose = np.loadtxt(str(pose_path)).reshape(4, 4)
        poses_c2w.append(pose)
    
    # Save trajectory file (all poses as flat 16-float rows)
    traj_path = output_base / "trajectory.txt"
    traj_data = np.stack(poses_c2w, axis=0)  # [N, 4, 4]
    np.savetxt(str(traj_path), traj_data.reshape(-1, 16), fmt='%.10e')
    
    print(f"Created {symlink_count} symlinks in {rgb_dir}")
    print(f"Saved trajectory ({len(poses_c2w)} poses) to {traj_path}")
    print(f"\nNext step: extract features")
    print(f"  CUDA_VISIBLE_DEVICES=0 python scripts/extract_multiscale_features.py \\")
    print(f"    --input_dir {rgb_dir} \\")
    print(f"    --output_dir {output_base}")

if __name__ == '__main__':
    main()
