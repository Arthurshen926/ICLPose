#!/usr/bin/env python3
"""
7-Scenes → ICLPose 格式转换脚本
==================================
将 7-Scenes 数据集（stairs 等场景）转换为本项目期望的 Replica 格式，
无需修改任何数据加载代码，也不影响正在运行的主进程。

7-Scenes 原始格式：
  {scene}/seq-NN/frame-NNNNNN.color.png     # RGB 图像 640×480
  {scene}/seq-NN/frame-NNNNNN.depth.png     # 深度图（16-bit，单位毫米）
  {scene}/seq-NN/frame-NNNNNN.pose.txt      # 4×4 Camera-to-World 矩阵（科学计数法）
  {scene}/TrainSplit.txt                    # 训练序列列表
  {scene}/TestSplit.txt                     # 测试序列列表
  {scene}/sparse/0/cameras.bin             # COLMAP 稀疏重建（可直接用于 vanilla 3DGS）

目标格式（项目期望）：
  {output}/{scene_name}/rgb/frame_NNNNNN.png   # 符号链接或复制
  {output}/{scene_name}/depth/depth_NNNNNN.png # 符号链接或复制
  {output}/{scene_name}/traj_w_c.txt           # 每行 16 个浮点，C2W 矩阵（行优先）

相机内参 (Kinect, SIMPLE_RADIAL, 从 COLMAP cameras.bin 读取)：
  fx = fy = 525.505, cx = 320.0, cy = 240.0

用法：
  python scripts/data_prep/convert_7scenes.py \\
      --src /mnt/pool1/sqy/7scenes/stairs \\
      --dst /home/yons/Projects/ICLPose/dataset/stairs \\
      --mode symlink   # symlink / copy
  
  转换后目录结构：
    dataset/stairs/seq-01/rgb/frame_000000.png → traj_w_c.txt
    dataset/stairs/seq-02/rgb/...
    ...
"""

import os
import re
import sys
import shutil
import struct
import argparse
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 常量：7-Scenes Kinect 内参（从 COLMAP cameras.bin 提取）
# ─────────────────────────────────────────────────────────────────────────────
SEVEN_SCENES_FX = 525.505
SEVEN_SCENES_FY = 525.505
SEVEN_SCENES_CX = 320.0
SEVEN_SCENES_CY = 240.0
SEVEN_SCENES_W = 640
SEVEN_SCENES_H = 480


def read_pose_txt(path: str) -> np.ndarray:
    """
    解析 7-Scenes 的 frame-NNNNNN.pose.txt。

    该文件为 4×4 C2W 矩阵，但每行用科学计数法+大量空格写成，
    实际上可能跨越多行（每个数字独占一行或拼接在同一行）。
    策略：读取全部 token，合并后取前 16 个浮点数。
    """
    with open(path, 'r') as f:
        content = f.read()
    tokens = content.split()
    if len(tokens) < 16:
        raise ValueError(f"位姿文件 token 不足 16: {path}  (got {len(tokens)})")
    values = [float(t) for t in tokens[:16]]
    pose = np.array(values, dtype=np.float64).reshape(4, 4)
    return pose


def convert_sequence(seq_dir: str, out_dir: str, mode: str = "symlink") -> int:
    """
    转换单个序列，返回转换的帧数。

    seq_dir: 如 /mnt/pool1/sqy/7scenes/stairs/seq-01
    out_dir: 如 dataset/stairs/seq-01
    mode:    "symlink" 或 "copy"
    """
    seq_dir = Path(seq_dir)
    out_dir = Path(out_dir)

    rgb_out = out_dir / "rgb"
    depth_out = out_dir / "depth"
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    # 枚举所有 frame-NNNNNN.color.png，按帧号排序
    color_files = sorted(seq_dir.glob("frame-*.color.png"))
    if not color_files:
        print(f"  [WARN] 未找到 color 文件: {seq_dir}")
        return 0

    poses = []
    for color_file in color_files:
        stem = color_file.stem.replace(".color", "")  # frame-000000
        m = re.match(r'frame-(\d+)', stem)
        if not m:
            continue
        idx = int(m.group(1))

        # ── RGB ──
        dst_rgb = rgb_out / f"frame_{idx:06d}.png"
        depth_file = seq_dir / f"{stem}.depth.png"
        dst_depth = depth_out / f"depth_{idx:06d}.png"
        pose_file = seq_dir / f"{stem}.pose.txt"

        if not pose_file.exists():
            print(f"  [WARN] 缺少位姿文件: {pose_file}")
            continue

        # 读取位姿
        pose = read_pose_txt(str(pose_file))
        poses.append(pose)

        # 创建 RGB 链接/复制
        if not dst_rgb.exists():
            if mode == "symlink":
                dst_rgb.symlink_to(color_file.resolve())
            else:
                shutil.copy2(color_file, dst_rgb)

        # 创建 Depth 链接/复制（如果存在）
        if depth_file.exists() and not dst_depth.exists():
            if mode == "symlink":
                dst_depth.symlink_to(depth_file.resolve())
            else:
                shutil.copy2(depth_file, dst_depth)

    # 写入 traj_w_c.txt（每行 16 个浮点，C2W 行优先）
    traj_path = out_dir / "traj_w_c.txt"
    with open(traj_path, 'w') as f:
        for pose in poses:
            row = " ".join(f"{v:.18e}" for v in pose.flatten())
            f.write(row + "\n")

    print(f"  [OK] {seq_dir.name}: {len(poses)} 帧 → {out_dir}")
    return len(poses)


def read_split_txt(path: str) -> list:
    """读取 TrainSplit.txt / TestSplit.txt，返回序列名称列表（如 'sequence1' → 'seq-01'）"""
    seqs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: "sequence1" → "seq-01"
            m = re.match(r'sequence(\d+)', line, re.IGNORECASE)
            if m:
                n = int(m.group(1))
                seqs.append(f"seq-{n:02d}")
            else:
                seqs.append(line)
    return seqs


def write_camera_intrinsics(out_dir: Path):
    """将相机内参写入 camera_intrinsics.txt 方便后续使用"""
    camera_file = out_dir / "camera_intrinsics.txt"
    with open(camera_file, 'w') as f:
        f.write("# 7-Scenes Kinect Camera Intrinsics\n")
        f.write("# SIMPLE_RADIAL model from COLMAP cameras.bin\n")
        f.write(f"# image_size: {SEVEN_SCENES_W} x {SEVEN_SCENES_H}\n")
        f.write(f"fx: {SEVEN_SCENES_FX}\n")
        f.write(f"fy: {SEVEN_SCENES_FY}\n")
        f.write(f"cx: {SEVEN_SCENES_CX}\n")
        f.write(f"cy: {SEVEN_SCENES_CY}\n")
        f.write(f"width: {SEVEN_SCENES_W}\n")
        f.write(f"height: {SEVEN_SCENES_H}\n")
        f.write("# Note: radial distortion k1=-0.0257987, ignored for simplicity\n")


def main():
    parser = argparse.ArgumentParser(
        description="将 7-Scenes 数据集转换为 ICLPose 项目所需格式"
    )
    parser.add_argument("--src", required=True,
                        help="7-Scenes 场景根目录，如 /mnt/pool1/sqy/7scenes/stairs")
    parser.add_argument("--dst", required=True,
                        help="输出目录，如 dataset/stairs")
    parser.add_argument("--seqs", nargs="*", default=None,
                        help="要转换的序列，如 seq-01 seq-02；不指定则转换所有序列")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="按 TrainSplit / TestSplit 过滤序列（默认转换全部）")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                        help="symlink: 创建符号链接（快，节省空间）; copy: 硬复制")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"[ERROR] 源目录不存在: {src}")
        sys.exit(1)

    # ── 确定要转换的序列 ──
    if args.seqs:
        target_seqs = args.seqs
    elif args.split != "all":
        split_file = src / ("TrainSplit.txt" if args.split == "train" else "TestSplit.txt")
        if not split_file.exists():
            print(f"[ERROR] 找不到分割文件: {split_file}")
            sys.exit(1)
        target_seqs = read_split_txt(str(split_file))
        print(f"[INFO] 从 {split_file.name} 读取到 {len(target_seqs)} 个序列: {target_seqs}")
    else:
        # 枚举所有 seq-NN 目录
        target_seqs = sorted([d.name for d in src.iterdir() if d.is_dir() and d.name.startswith("seq-")])
        print(f"[INFO] 发现 {len(target_seqs)} 个序列: {target_seqs}")

    if not target_seqs:
        print("[ERROR] 没有找到任何序列")
        sys.exit(1)

    # ── 写入场景级别的相机内参 ──
    dst.mkdir(parents=True, exist_ok=True)
    write_camera_intrinsics(dst)

    # ── 转换每个序列 ──
    total_frames = 0
    for seq_name in target_seqs:
        seq_dir = src / seq_name
        if not seq_dir.exists():
            print(f"  [WARN] 序列目录不存在，跳过: {seq_dir}")
            continue
        out_seq_dir = dst / seq_name
        n = convert_sequence(str(seq_dir), str(out_seq_dir), mode=args.mode)
        total_frames += n

    print(f"\n[完成] 共转换 {len(target_seqs)} 个序列，{total_frames} 帧")
    print(f"[提示] 输出目录: {dst}")
    print(f"[提示] 相机内参: fx={SEVEN_SCENES_FX}, fy={SEVEN_SCENES_FY}, "
          f"cx={SEVEN_SCENES_CX}, cy={SEVEN_SCENES_CY}")
    print()
    print("[下一步] 在训练配置中使用以下设置：")
    print(f"  dataset:")
    print(f"    data_root: \"{dst}\"")
    print(f"    train_scene: \"seq-02\"   # 根据 TrainSplit.txt 选择")
    print(f"    val_scene: \"seq-03\"")
    print(f"  camera:")
    print(f"    fx: {SEVEN_SCENES_FX}")
    print(f"    fy: {SEVEN_SCENES_FY}")
    print(f"    cx: {SEVEN_SCENES_CX}")
    print(f"    cy: {SEVEN_SCENES_CY}")
    print(f"    width: {SEVEN_SCENES_W}")
    print(f"    height: {SEVEN_SCENES_H}")


if __name__ == "__main__":
    main()
