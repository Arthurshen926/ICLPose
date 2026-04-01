#!/usr/bin/env python3
"""Convert Replica dataset format to COLMAP binary format.

Creates the sparse model (cameras.bin, images.bin, points3D.bin), symlinks
to RGB images, and a list_test.txt so that 2DGS-style training pipelines can
consume the result with ``--source_dir``.

Usage
-----
    python radio_gs/scripts/replica_to_colmap.py \
        --scene room_1 \
        --dataset_root dataset \
        --output_dir dataset/room_1
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Replica camera intrinsics (640×480)
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 640, 480
FX, FY = 320.0, 320.0
CX, CY = 319.5, 239.5

COLMAP_PINHOLE_MODEL_ID = 1  # PINHOLE in COLMAP

NUM_FRAMES_PER_SEQ = 900


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a COLMAP quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    candidates = np.array([
        trace,
        R[0, 0] - R[1, 1] - R[2, 2],
        -R[0, 0] + R[1, 1] - R[2, 2],
        -R[0, 0] - R[1, 1] + R[2, 2],
    ])
    i = np.argmax(candidates)

    if i == 0:
        s = 0.5 / np.sqrt(1.0 + trace)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif i == 1:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif i == 2:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    qvec = np.array([w, x, y, z], dtype=np.float64)
    # Ensure w >= 0 for canonical form
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# ---------------------------------------------------------------------------
# COLMAP binary writers
# ---------------------------------------------------------------------------

def write_cameras_bin(path: Path) -> None:
    """Write cameras.bin with a single PINHOLE camera."""
    with open(path, "wb") as f:
        # Number of cameras
        f.write(struct.pack("<Q", 1))
        # camera_id (uint32), model_id (int32), width (uint64), height (uint64)
        f.write(struct.pack("<iI", 1, COLMAP_PINHOLE_MODEL_ID))
        f.write(struct.pack("<QQ", IMG_W, IMG_H))
        # PINHOLE params: fx, fy, cx, cy
        f.write(struct.pack("<4d", FX, FY, CX, CY))


def write_images_bin(path: Path, image_entries: list[dict]) -> None:
    """Write images.bin.

    Each entry: {image_id, qvec, tvec, camera_id, name}.
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(image_entries)))
        for e in image_entries:
            # image_id (uint32)
            f.write(struct.pack("<I", e["image_id"]))
            # qvec: w, x, y, z (4 doubles)
            f.write(struct.pack("<4d", *e["qvec"]))
            # tvec: tx, ty, tz (3 doubles)
            f.write(struct.pack("<3d", *e["tvec"]))
            # camera_id (uint32)
            f.write(struct.pack("<I", e["camera_id"]))
            # image name (null-terminated string)
            name_bytes = e["name"].encode("utf-8") + b"\x00"
            f.write(name_bytes)
            # num_points2D = 0 (uint64)
            f.write(struct.pack("<Q", 0))


def write_points3d_bin(path: Path) -> None:
    """Write an empty points3D.bin."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------

def load_poses(traj_path: Path) -> np.ndarray:
    """Load traj_w_c.txt → (N, 4, 4) c2w matrices."""
    data = np.loadtxt(str(traj_path))
    poses = data.reshape(-1, 4, 4)
    return poses


# ---------------------------------------------------------------------------
# Symlink helpers
# ---------------------------------------------------------------------------

def make_image_symlinks(
    images_dir: Path,
    sequence_dir: Path,
    seq_name: str,
) -> list[str]:
    """Create symlinks under images_dir pointing to actual RGB PNGs.

    Returns the list of relative image names (as stored in images.bin).
    """
    rgb_src = sequence_dir / "rgb"
    if not rgb_src.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_src}")

    dst_seq_dir = images_dir / seq_name / "rgb"
    dst_seq_dir.mkdir(parents=True, exist_ok=True)

    # Collect sorted image files
    png_files = sorted(rgb_src.glob("rgb_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    names: list[str] = []
    for png in png_files:
        link = dst_seq_dir / png.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(png.resolve())
        # Relative name stored in COLMAP images.bin
        names.append(f"{seq_name}/rgb/{png.name}")
    return names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Replica dataset to COLMAP binary format."
    )
    parser.add_argument("--scene", required=True, help="Scene name, e.g. room_1")
    parser.add_argument(
        "--dataset_root", default="dataset", help="Root directory containing scenes"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: {dataset_root}/{scene})",
    )
    parser.add_argument(
        "--train_seq", default="Sequence_1", help="Training sequence name"
    )
    parser.add_argument(
        "--test_seq", default="Sequence_2", help="Test sequence name"
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    scene_dir = dataset_root / args.scene
    output_dir = Path(args.output_dir) if args.output_dir else scene_dir

    if not scene_dir.is_dir():
        print(f"Error: scene directory not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    # Prepare output directories
    sparse_dir = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    image_entries: list[dict] = []
    test_image_names: list[str] = []
    image_id = 1  # COLMAP image IDs are 1-indexed

    for seq_name, is_test in [(args.train_seq, False), (args.test_seq, True)]:
        seq_dir = scene_dir / seq_name
        traj_path = seq_dir / "traj_w_c.txt"

        if not traj_path.is_file():
            print(f"Warning: trajectory not found, skipping: {traj_path}", file=sys.stderr)
            continue

        # Load c2w poses and create symlinks
        c2w_poses = load_poses(traj_path)
        img_names = make_image_symlinks(images_dir, seq_dir, seq_name)

        if len(img_names) != len(c2w_poses):
            print(
                f"Warning: {seq_name} has {len(img_names)} images but "
                f"{len(c2w_poses)} poses — using min of both",
                file=sys.stderr,
            )

        n = min(len(img_names), len(c2w_poses))

        for i in range(n):
            c2w = c2w_poses[i]
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = rotmat_to_qvec(R)

            entry = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": t,
                "camera_id": 1,
                "name": img_names[i],
            }
            image_entries.append(entry)

            if is_test:
                test_image_names.append(img_names[i])

            image_id += 1

    # Write COLMAP binary files
    write_cameras_bin(sparse_dir / "cameras.bin")
    write_images_bin(sparse_dir / "images.bin", image_entries)
    write_points3d_bin(sparse_dir / "points3D.bin")

    # Write test image list
    list_test_path = sparse_dir / "list_test.txt"
    with open(list_test_path, "w") as f:
        for name in test_image_names:
            f.write(name + "\n")

    # Summary
    n_train = len(image_entries) - len(test_image_names)
    n_test = len(test_image_names)
    print(f"Scene:      {args.scene}")
    print(f"Output:     {output_dir}")
    print(f"Train:      {n_train} images")
    print(f"Test:       {n_test} images")
    print(f"Sparse:     {sparse_dir}")
    print(f"Images:     {images_dir}")
    print(f"list_test:  {list_test_path}")
    print("Done.")


if __name__ == "__main__":
    main()
