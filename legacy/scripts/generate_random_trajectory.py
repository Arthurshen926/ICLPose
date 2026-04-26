#!/usr/bin/env python3
"""
随机平滑轨迹生成器

在场景边界内生成随机但平滑的相机轨迹，用于合成数据生成。

使用方法:
    python scripts/generate_random_trajectory.py \
        --reference_traj dataset/room_0/Sequence_1/traj_tum.txt \
        --output_file dataset/room_0/Synthetic_Train/traj_tum.txt \
        --num_frames 1000
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicSpline
from tqdm import tqdm


def load_poses_tum(tum_file):
    """加载TUM格式的位姿文件"""
    poses = []
    
    with open(tum_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 8:
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                rot = R.from_quat([qx, qy, qz, qw])
                pose = np.eye(4)
                pose[:3, :3] = rot.as_matrix()
                pose[:3, 3] = [tx, ty, tz]
                
                poses.append(pose)
    
    return np.array(poses)


def get_scene_bounds(poses, margin=0.2):
    """从参考轨迹获取场景边界"""
    positions = poses[:, :3, 3]
    
    min_bounds = positions.min(axis=0) - margin
    max_bounds = positions.max(axis=0) + margin
    
    return min_bounds, max_bounds


def get_orientation_stats(poses):
    """获取参考轨迹的朝向统计信息"""
    rotations = [R.from_matrix(p[:3, :3]) for p in poses]
    eulers = np.array([r.as_euler('xyz') for r in rotations])
    
    return {
        'mean': eulers.mean(axis=0),
        'std': eulers.std(axis=0),
        'min': eulers.min(axis=0),
        'max': eulers.max(axis=0)
    }


def generate_smooth_path(num_waypoints, min_bounds, max_bounds, seed=None):
    """生成平滑的随机路径点"""
    if seed is not None:
        np.random.seed(seed)
    
    # 在边界内随机生成路径点
    waypoints = np.random.uniform(
        min_bounds, max_bounds, 
        size=(num_waypoints, 3)
    )
    
    return waypoints


def interpolate_positions(waypoints, num_frames):
    """使用三次样条插值生成平滑位置"""
    num_waypoints = len(waypoints)
    t_waypoints = np.linspace(0, 1, num_waypoints)
    t_frames = np.linspace(0, 1, num_frames)
    
    # 三次样条插值
    cs_x = CubicSpline(t_waypoints, waypoints[:, 0])
    cs_y = CubicSpline(t_waypoints, waypoints[:, 1])
    cs_z = CubicSpline(t_waypoints, waypoints[:, 2])
    
    positions = np.column_stack([
        cs_x(t_frames),
        cs_y(t_frames),
        cs_z(t_frames)
    ])
    
    return positions


def compute_look_at_rotations(positions, up=np.array([0, 0, 1])):
    """计算朝向下一个位置的旋转"""
    rotations = []
    
    for i in range(len(positions)):
        # 计算前进方向
        if i < len(positions) - 1:
            forward = positions[i + 1] - positions[i]
        else:
            forward = positions[i] - positions[i - 1]
        
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        # 计算右向量
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # forward几乎与up平行，使用备选up向量
            up_alt = np.array([0, 1, 0])
            right = np.cross(forward, up_alt)
            right_norm = np.linalg.norm(right)
        right = right / (right_norm + 1e-8)
        
        # 重新计算up向量
        up_new = np.cross(right, forward)
        up_new = up_new / (np.linalg.norm(up_new) + 1e-8)
        
        # 构建旋转矩阵 (相机坐标系: z向前, y向上, x向右)
        rot_mat = np.column_stack([right, -up_new, forward])
        rotations.append(R.from_matrix(rot_mat))
    
    return rotations


def smooth_rotations(rotations, window_size=5):
    """平滑旋转序列"""
    n = len(rotations)
    smoothed = []
    
    for i in range(n):
        # 获取窗口内的旋转
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        
        # 使用窗口中心的旋转作为平滑结果
        # 简化处理：直接使用当前旋转
        smoothed.append(rotations[i])
    
    return smoothed


def add_random_orientation_variation(rotations, angle_std_deg=5.0, seed=None):
    """添加随机的朝向变化"""
    if seed is not None:
        np.random.seed(seed)
    
    varied = []
    for rot in rotations:
        # 随机扰动 (小角度)
        delta_euler = np.random.normal(0, np.radians(angle_std_deg), 3)
        delta_rot = R.from_euler('xyz', delta_euler)
        varied.append(rot * delta_rot)
    
    return varied


def save_trajectory_tum(output_file, positions, rotations):
    """保存为TUM格式"""
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        
        for i, (pos, rot) in enumerate(zip(positions, rotations)):
            quat = rot.as_quat()  # [x, y, z, w]
            f.write(f"rgb_{i} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")
    
    print(f"✓ 轨迹已保存: {output_file}")
    print(f"  共 {len(positions)} 帧")


def visualize_trajectory(positions, reference_positions=None, output_file=None):
    """可视化轨迹 (可选)"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D视图
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                 'b-', label='Generated', linewidth=1)
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                    c='g', s=100, marker='o', label='Start')
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                    c='r', s=100, marker='x', label='End')
        
        if reference_positions is not None:
            ax1.plot(reference_positions[:, 0], reference_positions[:, 1], 
                     reference_positions[:, 2], 'gray', alpha=0.5, 
                     label='Reference', linewidth=0.5)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        ax1.set_title('3D Trajectory')
        
        # 俯视图 (XY)
        ax2 = fig.add_subplot(122)
        ax2.plot(positions[:, 0], positions[:, 1], 'b-', label='Generated', linewidth=1)
        ax2.scatter(positions[0, 0], positions[0, 1], c='g', s=100, marker='o', label='Start')
        ax2.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, marker='x', label='End')
        
        if reference_positions is not None:
            ax2.plot(reference_positions[:, 0], reference_positions[:, 1], 
                     'gray', alpha=0.5, label='Reference', linewidth=0.5)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.axis('equal')
        ax2.legend()
        ax2.set_title('Top View (XY)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✓ 轨迹可视化已保存: {output_file}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("警告: matplotlib未安装，跳过可视化")


def main():
    parser = argparse.ArgumentParser(description='生成随机平滑轨迹')
    parser.add_argument('--reference_traj', type=str, required=True,
                        help='参考轨迹文件 (用于确定场景边界)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出轨迹文件')
    parser.add_argument('--num_frames', type=int, default=1000,
                        help='生成的帧数')
    parser.add_argument('--num_waypoints', type=int, default=20,
                        help='路径控制点数量')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='场景边界外扩余量')
    parser.add_argument('--angle_variation', type=float, default=5.0,
                        help='朝向随机变化标准差 (度)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化轨迹')
    parser.add_argument('--vis_output', type=str, default=None,
                        help='可视化输出文件')
    args = parser.parse_args()
    
    print("=" * 60)
    print("随机平滑轨迹生成")
    print("=" * 60)
    
    # 加载参考轨迹
    print(f"\n加载参考轨迹: {args.reference_traj}")
    ref_poses = load_poses_tum(args.reference_traj)
    print(f"  参考帧数: {len(ref_poses)}")
    
    # 获取场景边界
    min_bounds, max_bounds = get_scene_bounds(ref_poses, args.margin)
    print(f"\n场景边界:")
    print(f"  X: [{min_bounds[0]:.2f}, {max_bounds[0]:.2f}]")
    print(f"  Y: [{min_bounds[1]:.2f}, {max_bounds[1]:.2f}]")
    print(f"  Z: [{min_bounds[2]:.2f}, {max_bounds[2]:.2f}]")
    
    # 生成随机路径点
    print(f"\n生成 {args.num_waypoints} 个路径控制点...")
    waypoints = generate_smooth_path(
        args.num_waypoints, min_bounds, max_bounds, args.seed
    )
    
    # 插值生成平滑位置
    print(f"三次样条插值生成 {args.num_frames} 帧...")
    positions = interpolate_positions(waypoints, args.num_frames)
    
    # 计算朝向 (look-at)
    print("计算相机朝向...")
    rotations = compute_look_at_rotations(positions)
    
    # 添加朝向变化
    if args.angle_variation > 0:
        print(f"添加朝向随机变化 (std={args.angle_variation}°)...")
        seed_rot = args.seed + 1 if args.seed else None
        rotations = add_random_orientation_variation(
            rotations, args.angle_variation, seed_rot
        )
    
    # 保存轨迹
    print(f"\n保存轨迹...")
    save_trajectory_tum(args.output_file, positions, rotations)
    
    # 可视化
    if args.visualize or args.vis_output:
        print("\n生成可视化...")
        ref_positions = ref_poses[:, :3, 3]
        vis_output = args.vis_output
        if vis_output is None and args.visualize:
            vis_output = str(Path(args.output_file).with_suffix('.png'))
        visualize_trajectory(positions, ref_positions, vis_output)
    
    print("\n✓ 完成!")


if __name__ == '__main__':
    main()
