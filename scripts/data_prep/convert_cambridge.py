#!/usr/bin/env python3
"""
Cambridge Landmarks → ICLPose 格式转换脚本
===========================================
将 Cambridge Landmarks 数据集（OldHospital、GreatCourt 等）转换为本项目期望格式，
不改动任何现有数据加载代码。

Cambridge 原始格式：
  {scene}/seqN/frameXXXXX.png            # RGB 图像 1920×1080（室外）
  {scene}/dataset_train.txt              # 训练集位姿文件
  {scene}/dataset_test.txt              # 测试集位姿文件
  {scene}/reconstruction.nvm            # VisualSFM NVM 稀疏重建（可转 COLMAP 供 3DGS 用）

dataset_train.txt 格式（前两行为注释）：
  seqN/frameXXXXX.png  X  Y  Z  qw  qx  qy  qz
  # X Y Z = 相机中心坐标（世界系）
  # qw qx qy qz = Camera-to-World 旋转四元数（标量部分在前）

目标格式（项目期望）：
  {output}/{split}/rgb/frame_XXXXXX.png    # 符号链接或复制
  {output}/{split}/traj_w_c.txt            # 每行 16 个浮点，C2W 矩阵（行优先）

相机内参（从 NVM focal length 解析）：
  OldHospital: fx = fy ≈ 1673.27, cx = 960.0, cy = 540.0  (1920×1080)

用法：
  python scripts/data_prep/convert_cambridge.py \\
      --src dataset/OldHospital \\
      --dst dataset/cambridge_OldHospital \\
      --mode symlink

  # 转换后：
  #   dataset/cambridge_OldHospital/train/rgb/  + traj_w_c.txt
  #   dataset/cambridge_OldHospital/test/rgb/   + traj_w_c.txt
"""

import os
import re
import sys
import shutil
import argparse
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# NVM 内参解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_nvm_focal(nvm_path: str) -> float:
    """
    从 NVM 文件读取第一个相机的焦距（单位：像素）。
    NVM 格式（相机行）：
      filename  focal_len  qw qx qy qz  tx ty tz  radial  flag
    """
    with open(nvm_path, 'r', errors='replace') as f:
        lines = f.readlines()

    # 找到相机数量行
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('N'):
            continue
        try:
            num_cams = int(line)
            # 下一行即第一个相机
            if i + 1 < len(lines):
                parts = lines[i + 1].strip().split()
                return float(parts[1])
        except ValueError:
            continue
    return None


def parse_cambridge_pose_file(pose_txt: str, src_root: Path):
    """
    解析 Cambridge 的 dataset_train.txt 或 dataset_test.txt。

    返回：list of (image_path_abs, pose_c2w_4x4)
      - image_path_abs: 原始图像的绝对路径
      - pose_c2w_4x4: 4×4 Camera-to-World 矩阵 (float64)
    """
    records = []
    with open(pose_txt, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('I'):
            continue  # 跳过注释和头部
        parts = line.split()
        if len(parts) < 8:
            continue
        rel_path = parts[0]           # seqN/frameXXXXX.png
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

        # 四元数 → 旋转矩阵（C2W，即世界坐标系下的相机旋转）
        # 使用 numpy 手算，避免额外依赖
        R = quat2rot(qw, qx, qy, qz)

        # C2W 矩阵
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        img_abs = src_root / rel_path
        records.append((img_abs, T, rel_path))

    return records


def quat2rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """单位四元数 (qw, qx, qy, qz) → 3×3 旋转矩阵"""
    # 归一化
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,   2*qx*qy - 2*qz*qw,   2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,   1 - 2*qx**2 - 2*qz**2,   2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,   2*qy*qz + 2*qx*qw,   1 - 2*qx**2 - 2*qy**2],
    ], dtype=np.float64)
    return R


def convert_split(records, src_root: Path, out_dir: Path, mode: str):
    """将一个分割（train/test）的所有记录写入目标目录"""
    rgb_out = out_dir / "rgb"
    rgb_out.mkdir(parents=True, exist_ok=True)

    poses = []
    created = 0
    skipped = 0
    for idx, (img_abs, pose, rel_path) in enumerate(records):
        if not img_abs.exists():
            print(f"  [WARN] 图像文件不存在，跳过: {img_abs}")
            skipped += 1
            continue

        # 用全局索引生成目标文件名，避免不同 seq 的 frame 编号冲突
        # 保留原始 seq+frame 信息以便追溯
        seq_match = re.match(r'seq(\d+)/frame(\d+)', rel_path)
        if seq_match:
            seq_n = int(seq_match.group(1))
            frame_n = int(seq_match.group(2))
            dst_name = f"seq{seq_n:02d}_frame{frame_n:06d}.png"
        else:
            dst_name = f"frame_{idx:06d}.png"

        dst_rgb = rgb_out / dst_name
        if not dst_rgb.exists():
            if mode == "symlink":
                dst_rgb.symlink_to(img_abs.resolve())
            else:
                shutil.copy2(img_abs, dst_rgb)

        poses.append(pose)
        created += 1

    # 写 traj_w_c.txt
    traj_path = out_dir / "traj_w_c.txt"
    with open(traj_path, 'w') as f:
        for pose in poses:
            row = " ".join(f"{v:.18e}" for v in pose.flatten())
            f.write(row + "\n")

    print(f"  [OK] {out_dir.name}: {created} 帧写入 (跳过 {skipped})")
    return created


def write_camera_intrinsics(out_dir: Path, fx: float, fy: float,
                             cx: float, cy: float, w: int, h: int):
    camera_file = out_dir / "camera_intrinsics.txt"
    with open(camera_file, 'w') as f:
        f.write("# Cambridge Landmarks Camera Intrinsics\n")
        f.write("# Focal length extracted from reconstruction.nvm\n")
        f.write(f"# image_size: {w} x {h}\n")
        f.write(f"fx: {fx:.6f}\n")
        f.write(f"fy: {fy:.6f}\n")
        f.write(f"cx: {cx:.6f}\n")
        f.write(f"cy: {cy:.6f}\n")
        f.write(f"width: {w}\n")
        f.write(f"height: {h}\n")


def main():
    parser = argparse.ArgumentParser(
        description="将 Cambridge Landmarks 数据集转换为 ICLPose 项目所需格式"
    )
    parser.add_argument("--src", required=True,
                        help="Cambridge 场景根目录，如 dataset/OldHospital")
    parser.add_argument("--dst", required=True,
                        help="输出目录，如 dataset/cambridge_OldHospital")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                        help="symlink: 符号链接（快，省空间）; copy: 硬复制")
    parser.add_argument("--width", type=int, default=1920, help="图像宽度（默认 1920）")
    parser.add_argument("--height", type=int, default=1080, help="图像高度（默认 1080）")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"[ERROR] 源目录不存在: {src}")
        sys.exit(1)

    # ── 解析 NVM 内参 ──
    nvm_path = src / "reconstruction.nvm"
    focal = None
    if nvm_path.exists():
        focal = parse_nvm_focal(str(nvm_path))
        print(f"[INFO] 从 NVM 解析焦距: {focal:.2f} px")
    if focal is None:
        focal = 1673.27  # OldHospital 默认值
        print(f"[WARN] 无法解析 NVM 焦距，使用默认值 {focal}")

    cx = args.width / 2.0
    cy = args.height / 2.0

    dst.mkdir(parents=True, exist_ok=True)
    write_camera_intrinsics(dst, focal, focal, cx, cy, args.width, args.height)

    total = 0
    for split_name, txt_name in [("train", "dataset_train.txt"), ("test", "dataset_test.txt")]:
        txt_path = src / txt_name
        if not txt_path.exists():
            print(f"  [SKIP] {txt_name} 不存在，跳过 {split_name}")
            continue

        print(f"\n[INFO] 处理 {split_name} split: {txt_path}")
        records = parse_cambridge_pose_file(str(txt_path), src)
        print(f"  解析到 {len(records)} 条记录")

        out_split = dst / split_name
        n = convert_split(records, src, out_split, mode=args.mode)
        total += n

    print(f"\n[完成] 共处理 {total} 帧")
    print(f"[提示] 输出目录: {dst}")
    print(f"[提示] 相机内参: fx=fy={focal:.2f}, cx={cx:.1f}, cy={cy:.1f}")
    print()
    print("[下一步] 在训练配置中使用以下设置：")
    print(f"  dataset:")
    print(f"    data_root: \"{dst}\"")
    print(f"    train_scene: \"train\"")
    print(f"    val_scene: \"test\"")
    print(f"  camera:")
    print(f"    fx: {focal:.2f}")
    print(f"    fy: {focal:.2f}")
    print(f"    cx: {cx:.1f}")
    print(f"    cy: {cy:.1f}")
    print(f"    width: {args.width}")
    print(f"    height: {args.height}")


if __name__ == "__main__":
    main()
