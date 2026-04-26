#!/usr/bin/env python3
"""
Convert Replica dataset format to COLMAP format for 2DGS training.

Replica format:
  - dataset/room_0/Sequence_1/rgb/rgb_0.png ... rgb_899.png
  - dataset/room_0/Sequence_1/traj_w_c.txt  (c2w 4x4 matrices, one per line)
  - dataset/room_0/Sequence_1/depth/depth_0.png ... (uint16, mm)
  - Known intrinsics: fx=fy=320, cx=319.5, cy=239.5, 640x480

Output COLMAP format:
  - sparse/0/cameras.bin, images.bin, points3D.ply
  - images/ (symlinks to original rgb)
  - dataset_train.txt, dataset_test.txt
"""

import argparse
import os
import struct
import sys

import numpy as np


def rotmat2qvec(R):
    """Rotation matrix -> COLMAP quaternion (w, x, y, z).

    Uses a numerically stable Shepperd-method implementation.
    The previous eigenvector-based implementation encoded R^T instead of R
    due to np.linalg.eigh treating the lower-triangular input as symmetric.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    qvec = np.array([qw, qx, qy, qz])
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cameras_binary(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(cameras)))
        for cam_id, (model_id, width, height, params) in cameras.items():
            f.write(struct.pack("I", cam_id))
            f.write(struct.pack("i", model_id))
            f.write(struct.pack("Q", width))
            f.write(struct.pack("Q", height))
            for p in params:
                f.write(struct.pack("d", p))


def write_images_binary(images, path):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(images)))
        for img_id, (qvec, tvec, cam_id, name) in images.items():
            f.write(struct.pack("I", img_id))
            for q in qvec:
                f.write(struct.pack("d", q))
            for t in tvec:
                f.write(struct.pack("d", t))
            f.write(struct.pack("I", cam_id))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("Q", 0))  # no 2D points


def create_point_cloud_from_depth(rgb_dir, depth_dir, poses_c2w, intrinsics,
                                  subsample=10, max_points=200000):
    """Create sparse point cloud from depth maps."""
    from PIL import Image
    fx, fy, cx, cy = intrinsics
    all_xyz = []
    all_rgb = []
    n_frames = len(poses_c2w)
    # Sample from every Nth frame
    step = max(1, n_frames // 50)
    for i in range(0, n_frames, step):
        depth_path = os.path.join(depth_dir, f"depth_{i}.png")
        rgb_path = os.path.join(rgb_dir, f"rgb_{i}.png")
        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue
        depth = np.array(Image.open(depth_path), dtype=np.float32)
        # Replica depth is in mm (uint16)
        if depth.max() > 100:
            depth = depth / 1000.0  # mm -> m
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        H, W = depth.shape[:2]
        # Subsample pixels
        ys, xs = np.mgrid[0:H:subsample, 0:W:subsample]
        ys, xs = ys.flatten(), xs.flatten()
        zs = depth[ys, xs]
        valid = (zs > 0.01) & (zs < 10.0)
        ys, xs, zs = ys[valid], xs[valid], zs[valid]
        # Back-project to 3D (camera coords)
        x3d = (xs - cx) / fx * zs
        y3d = (ys - cy) / fy * zs
        pts_cam = np.stack([x3d, y3d, zs], axis=-1)  # [N, 3]
        # Transform to world coords
        c2w = poses_c2w[i]
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        pts_world = (R @ pts_cam.T).T + t  # [N, 3]
        colors = rgb[ys, xs]
        all_xyz.append(pts_world)
        all_rgb.append(colors)
    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)
    # Random subsample if too many
    if len(all_xyz) > max_points:
        idx = np.random.choice(len(all_xyz), max_points, replace=False)
        all_xyz = all_xyz[idx]
        all_rgb = all_rgb[idx]
    return all_xyz.astype(np.float32), all_rgb.astype(np.float32)


def write_points3d_ply(xyz, rgb, path):
    """Write point cloud as PLY."""
    from plyfile import PlyData, PlyElement
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    vertices = np.empty(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertices['x'] = xyz[:, 0]
    vertices['y'] = xyz[:, 1]
    vertices['z'] = xyz[:, 2]
    vertices['red'] = rgb_uint8[:, 0]
    vertices['green'] = rgb_uint8[:, 1]
    vertices['blue'] = rgb_uint8[:, 2]
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--room_dir", required=True, help="e.g. dataset/room_0")
    parser.add_argument("--train_seq", default="Sequence_1")
    parser.add_argument("--test_seq", default="Sequence_2")
    parser.add_argument("--fx", type=float, default=320.0)
    parser.add_argument("--fy", type=float, default=320.0)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max_points", type=int, default=200000)
    args = parser.parse_args()

    out_dir = args.room_dir
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # --- Write cameras.bin (PINHOLE model_id=1) ---
    cameras = {1: (1, args.width, args.height, [args.fx, args.fy, args.cx, args.cy])}
    write_cameras_binary(cameras, os.path.join(sparse_dir, "cameras.bin"))
    print(f"Wrote cameras.bin: PINHOLE {args.width}x{args.height} fx={args.fx}")

    # --- Load poses and create images ---
    images = {}
    test_names = []
    img_id = 1
    all_poses_c2w = []

    for seq_name, is_test in [(args.train_seq, False), (args.test_seq, True)]:
        seq_dir = os.path.join(args.room_dir, seq_name)
        traj_path = os.path.join(seq_dir, "traj_w_c.txt")
        rgb_dir = os.path.join(seq_dir, "rgb")

        if not os.path.exists(traj_path):
            print(f"WARNING: {traj_path} not found, skipping {seq_name}")
            continue

        poses = np.loadtxt(traj_path).reshape(-1, 4, 4)  # c2w
        n_frames = len(poses)
        print(f"  {seq_name}: {n_frames} frames (test={is_test})")

        for i in range(n_frames):
            rgb_name = f"rgb_{i}.png"
            src = os.path.join(rgb_dir, rgb_name)
            # Use relative path: Sequence_X/rgb/rgb_Y.png
            colmap_name = f"{seq_name}/rgb/{rgb_name}"
            # Create symlink in images/
            dst = os.path.join(images_dir, seq_name, "rgb")
            os.makedirs(dst, exist_ok=True)
            dst_file = os.path.join(dst, rgb_name)
            if not os.path.exists(dst_file) and os.path.exists(src):
                os.symlink(os.path.abspath(src), dst_file)

            # Convert c2w to COLMAP w2c
            c2w = poses[i]
            R_c2w = c2w[:3, :3]
            t_c2w = c2w[:3, 3]
            # w2c: R_w2c = R_c2w^T, t_w2c = -R_c2w^T @ t_c2w
            R_w2c = R_c2w.T
            t_w2c = -R_c2w.T @ t_c2w
            qvec = rotmat2qvec(R_w2c)

            images[img_id] = (qvec, t_w2c, 1, colmap_name)
            if is_test:
                test_names.append(colmap_name)
            img_id += 1

            if not is_test:
                all_poses_c2w.append(c2w)

    write_images_binary(images, os.path.join(sparse_dir, "images.bin"))
    print(f"Wrote images.bin: {len(images)} images")

    # --- Write test list ---
    list_test_path = os.path.join(sparse_dir, "list_test.txt")
    with open(list_test_path, "w") as f:
        for name in test_names:
            f.write(name + "\n")
    print(f"Wrote list_test.txt: {len(test_names)} test images")

    # --- Create point cloud from depth ---
    train_seq_dir = os.path.join(args.room_dir, args.train_seq)
    depth_dir = os.path.join(train_seq_dir, "depth")
    rgb_dir = os.path.join(train_seq_dir, "rgb")

    if os.path.exists(depth_dir):
        print("Creating point cloud from depth maps...")
        all_poses_c2w = np.array(all_poses_c2w)
        xyz, rgb = create_point_cloud_from_depth(
            rgb_dir, depth_dir, all_poses_c2w,
            (args.fx, args.fy, args.cx, args.cy),
            subsample=10, max_points=args.max_points,
        )
        ply_path = os.path.join(sparse_dir, "points3D.ply")
        write_points3d_ply(xyz, rgb, ply_path)
        print(f"Wrote points3D.ply: {len(xyz):,} points")

        # Also write empty points3D.bin for compatibility
        with open(os.path.join(sparse_dir, "points3D.bin"), "wb") as f:
            f.write(struct.pack("Q", 0))
    else:
        print(f"WARNING: No depth dir at {depth_dir}, creating random point cloud")
        # Fallback: random points around camera centers
        cam_centers = []
        for c2w in all_poses_c2w:
            cam_centers.append(c2w[:3, 3])
        cam_centers = np.array(cam_centers)
        center = cam_centers.mean(axis=0)
        scale = np.max(np.linalg.norm(cam_centers - center, axis=1)) * 2
        xyz = center + np.random.randn(10000, 3).astype(np.float32) * scale * 0.3
        rgb = np.random.rand(10000, 3).astype(np.float32) * 0.5 + 0.25
        ply_path = os.path.join(sparse_dir, "points3D.ply")
        write_points3d_ply(xyz, rgb, ply_path)
        print(f"Wrote points3D.ply: {len(xyz):,} random points (fallback)")

    print(f"\nConversion complete! COLMAP data at: {sparse_dir}")
    print(f"Images symlinked at: {images_dir}")


if __name__ == "__main__":
    main()
