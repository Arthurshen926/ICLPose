#!/usr/bin/env python3
"""
从 PLY 点云文件自动计算场景边界 (scene bound)
用于填写训练配置中的 scene.bound 字段。

用法：
  python scripts/data_prep/compute_scene_bound.py \\
      --ply dataset/stairs/gaussian_splatting/point_cloud/final/point_cloud.ply \\
      --margin 0.5

输出：
  场景边界 [[x_min, x_max], [y_min, y_max], [z_min, z_max]] (含 margin)
  可以直接复制到 yaml 配置文件中
"""

import argparse
import numpy as np
from pathlib import Path


def compute_bound_from_ply(ply_path: str, margin: float = 0.5) -> dict:
    """读取 PLY 点云，返回带 margin 的轴对齐边界框"""
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("请安装 plyfile: pip install plyfile")

    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    pts = np.stack([v['x'], v['y'], v['z']], axis=1)

    # 剔除离群点（使用 1-99 百分位）
    p1, p99 = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    mask = np.all((pts >= p1) & (pts <= p99), axis=1)
    pts_clean = pts[mask]

    mins = pts_clean.min(axis=0) - margin
    maxs = pts_clean.max(axis=0) + margin

    return {
        'num_points': len(pts),
        'num_clean': int(mask.sum()),
        'bound': [[float(mins[i]), float(maxs[i])] for i in range(3)],
        'center': ((mins + maxs) / 2).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="计算 PLY 点云的场景边界")
    parser.add_argument("--ply", required=True, help="PLY 文件路径")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="边界扩展量（米），默认 0.5")
    args = parser.parse_args()

    if not Path(args.ply).exists():
        print(f"[ERROR] PLY 文件不存在: {args.ply}")
        raise SystemExit(1)

    result = compute_bound_from_ply(args.ply, args.margin)
    bound = result['bound']

    print(f"\n[INFO] 点云总数: {result['num_points']}")
    print(f"[INFO] 剔除离群后: {result['num_clean']} 点")
    print(f"\n场景边界（含 margin={args.margin}m）:")
    print(f"  x: [{bound[0][0]:.3f}, {bound[0][1]:.3f}]")
    print(f"  y: [{bound[1][0]:.3f}, {bound[1][1]:.3f}]")
    print(f"  z: [{bound[2][0]:.3f}, {bound[2][1]:.3f}]")
    print(f"\n[YAML 格式 - 直接粘贴到 configs/]:")
    print(f"scene:")
    print(f"  bound: [[{bound[0][0]:.3f}, {bound[0][1]:.3f}], "
          f"[{bound[1][0]:.3f}, {bound[1][1]:.3f}], "
          f"[{bound[2][0]:.3f}, {bound[2][1]:.3f}]]")


if __name__ == "__main__":
    main()
