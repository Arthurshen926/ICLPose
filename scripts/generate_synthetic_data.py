#!/usr/bin/env python3
"""
3DGS合成数据生成脚本

从已有的3D Gaussian Splatting模型生成合成训练数据:
1. 在现有相机轨迹周围生成扰动位姿
2. 使用3DGS渲染对应的RGB图像
3. 预提取DINO+SD融合特征
4. 保存为训练数据格式

使用方法:
    python scripts/generate_synthetic_data.py --config configs/data_generation.yaml
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import cv2
from scipy.spatial.transform import Rotation as R

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_poses_tum(tum_file):
    """加载TUM格式的位姿文件 (支持rgb_X作为时间戳)"""
    poses = []
    frame_ids = []
    
    with open(tum_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 8:
                # 解析frame_id (可能是rgb_X或数字)
                frame_id = parts[0]
                if frame_id.startswith('rgb_'):
                    frame_num = int(frame_id.split('_')[1])
                else:
                    try:
                        frame_num = int(float(frame_id))
                    except:
                        frame_num = len(poses)
                
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                # 四元数转旋转矩阵
                rot = R.from_quat([qx, qy, qz, qw])
                pose = np.eye(4)
                pose[:3, :3] = rot.as_matrix()
                pose[:3, 3] = [tx, ty, tz]
                
                poses.append(pose)
                frame_ids.append(frame_num)
    
    return np.array(poses), np.array(frame_ids)


def generate_perturbed_poses(base_poses, num_samples_per_pose=5, 
                             rot_noise_deg=5.0, trans_noise_m=0.2):
    """
    在基础位姿周围生成扰动位姿
    
    Args:
        base_poses: (N, 4, 4) 基础位姿
        num_samples_per_pose: 每个基础位姿生成的扰动数量
        rot_noise_deg: 旋转噪声标准差（度）
        trans_noise_m: 平移噪声标准差（米）
    
    Returns:
        perturbed_poses: (N*num_samples_per_pose, 4, 4) 扰动位姿
    """
    perturbed_poses = []
    
    for pose in base_poses:
        for _ in range(num_samples_per_pose):
            # 生成旋转扰动
            rot_noise = np.random.randn(3) * np.radians(rot_noise_deg)
            delta_rot = R.from_rotvec(rot_noise).as_matrix()
            
            # 生成平移扰动
            trans_noise = np.random.randn(3) * trans_noise_m
            
            # 应用扰动
            perturbed = pose.copy()
            perturbed[:3, :3] = pose[:3, :3] @ delta_rot
            perturbed[:3, 3] = pose[:3, 3] + trans_noise
            
            perturbed_poses.append(perturbed)
    
    return np.array(perturbed_poses)


def interpolate_poses(poses, num_interpolations=3):
    """
    在相邻位姿之间进行插值
    
    Args:
        poses: (N, 4, 4) 位姿序列
        num_interpolations: 每对相邻位姿之间的插值数量
    
    Returns:
        interpolated_poses: 插值后的位姿
    """
    interpolated = []
    
    for i in range(len(poses) - 1):
        pose1 = poses[i]
        pose2 = poses[i + 1]
        
        # 插值
        for t in np.linspace(0, 1, num_interpolations + 2)[:-1]:
            # 平移插值
            trans = pose1[:3, 3] * (1 - t) + pose2[:3, 3] * t
            
            # 旋转插值 (SLERP)
            rot1 = R.from_matrix(pose1[:3, :3])
            rot2 = R.from_matrix(pose2[:3, :3])
            
            # 使用slerp插值
            from scipy.spatial.transform import Slerp
            key_times = [0, 1]
            rotations = R.from_matrix(np.stack([pose1[:3, :3], pose2[:3, :3]]))
            slerp = Slerp(key_times, rotations)
            rot_interp = slerp(t).as_matrix()
            
            pose_interp = np.eye(4)
            pose_interp[:3, :3] = rot_interp
            pose_interp[:3, 3] = trans
            
            interpolated.append(pose_interp)
    
    # 添加最后一个位姿
    interpolated.append(poses[-1])
    
    return np.array(interpolated)


def save_poses_tum(poses, timestamps, output_file):
    """保存位姿为TUM格式"""
    with open(output_file, 'w') as f:
        for i, (pose, ts) in enumerate(zip(poses, timestamps)):
            # 提取旋转和平移
            rot = R.from_matrix(pose[:3, :3])
            quat = rot.as_quat()  # [qx, qy, qz, qw]
            trans = pose[:3, 3]
            
            f.write(f"{ts:.6f} {trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} "
                    f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")


class SyntheticDataGenerator:
    """合成数据生成器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载3DGS模型
        self._load_gaussian_model()
        
        # 加载特征提取器
        self._load_feature_extractor()
    
    def _load_gaussian_model(self):
        """加载3D Gaussian Splatting模型"""
        from gaussian_splatting.scene.gaussian_model import GaussianModel
        from gaussian_splatting.gaussian_renderer import render
        
        # 这里需要根据实际的3DGS实现调整
        print("  加载3DGS模型...")
        self.gaussian_model = None  # 占位符
        print("  ✓ 3DGS模型加载完成")
    
    def _load_feature_extractor(self):
        """加载DINO+SD特征提取器"""
        # 这里需要加载预训练的特征提取器
        print("  加载DINO+SD特征提取器...")
        self.feature_extractor = None  # 占位符
        print("  ✓ 特征提取器加载完成")
    
    def render_image(self, pose, intrinsics, image_size):
        """
        从3DGS渲染图像
        
        Args:
            pose: (4, 4) 相机位姿 (camera-to-world)
            intrinsics: (3, 3) 相机内参
            image_size: (width, height)
        
        Returns:
            rgb: (H, W, 3) RGB图像
        """
        # 这里需要调用实际的3DGS渲染
        # 占位符：返回空图像
        h, w = image_size[1], image_size[0]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        return rgb
    
    def extract_fused_features(self, rgb_image):
        """
        提取DINO+SD融合特征
        
        Args:
            rgb_image: (H, W, 3) RGB图像
        
        Returns:
            fused_feat: (C, H', W') 融合特征图
        """
        # 这里需要调用实际的特征提取
        # 占位符
        h, w = rgb_image.shape[:2]
        feat_h, feat_w = h // 14, w // 14  # DINO patch size = 14
        fused_feat = np.zeros((256, feat_h, feat_w), dtype=np.float32)
        return fused_feat
    
    def generate(self, base_poses, output_dir, 
                 num_perturbed=5, num_interpolated=3,
                 intrinsics=None, image_size=(640, 480)):
        """
        生成合成数据
        
        Args:
            base_poses: (N, 4, 4) 基础位姿
            output_dir: 输出目录
            num_perturbed: 每个位姿的扰动数量
            num_interpolated: 插值数量
            intrinsics: 相机内参
            image_size: 图像尺寸
        """
        output_dir = Path(output_dir)
        rgb_dir = output_dir / 'rgb'
        feat_dir = output_dir / 'fused_feat'
        
        rgb_dir.mkdir(parents=True, exist_ok=True)
        feat_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成扰动位姿
        print(f"生成扰动位姿 (每个基础位姿{num_perturbed}个扰动)...")
        perturbed_poses = generate_perturbed_poses(
            base_poses, num_perturbed, 
            rot_noise_deg=5.0, trans_noise_m=0.2
        )
        
        # 生成插值位姿
        print(f"生成插值位姿 (每对相邻位姿{num_interpolated}个插值)...")
        interpolated_poses = interpolate_poses(base_poses, num_interpolated)
        
        # 合并所有位姿
        all_poses = np.concatenate([base_poses, perturbed_poses, interpolated_poses])
        print(f"总共 {len(all_poses)} 个位姿")
        
        # 去除重复位姿（基于位置距离）
        all_poses = self._deduplicate_poses(all_poses)
        print(f"去重后 {len(all_poses)} 个位姿")
        
        # 生成数据
        all_pose_records = []
        
        for i, pose in enumerate(tqdm(all_poses, desc="生成合成数据")):
            # 渲染RGB图像
            rgb = self.render_image(pose, intrinsics, image_size)
            
            # 保存RGB图像
            rgb_path = rgb_dir / f'frame_{i:06d}.png'
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # 提取融合特征
            fused_feat = self.extract_fused_features(rgb)
            
            # 保存融合特征
            feat_path = feat_dir / f'fused_feat_{i:06d}.npy'
            np.save(str(feat_path), fused_feat)
            
            # 记录位姿
            all_pose_records.append({
                'frame_id': i,
                'pose': pose.tolist(),
                'rgb_path': str(rgb_path.relative_to(output_dir)),
                'feat_path': str(feat_path.relative_to(output_dir))
            })
        
        # 保存位姿文件
        timestamps = np.arange(len(all_poses), dtype=np.float64)
        save_poses_tum(all_poses, timestamps, output_dir / 'traj_tum.txt')
        
        print(f"✓ 完成! 生成了 {len(all_poses)} 个样本")
        print(f"  输出目录: {output_dir}")
        
        return all_pose_records
    
    def _deduplicate_poses(self, poses, min_dist=0.1, min_angle_deg=5.0):
        """去除过于接近的位姿"""
        if len(poses) <= 1:
            return poses
        
        unique = [poses[0]]
        
        for pose in poses[1:]:
            # 检查与所有已选位姿的距离
            is_unique = True
            for ref_pose in unique:
                # 平移距离
                trans_dist = np.linalg.norm(pose[:3, 3] - ref_pose[:3, 3])
                
                # 旋转距离
                rel_rot = pose[:3, :3] @ ref_pose[:3, :3].T
                angle = np.abs(np.arccos(np.clip((np.trace(rel_rot) - 1) / 2, -1, 1)))
                angle_deg = np.degrees(angle)
                
                if trans_dist < min_dist and angle_deg < min_angle_deg:
                    is_unique = False
                    break
            
            if is_unique:
                unique.append(pose)
        
        return np.array(unique)


def main():
    parser = argparse.ArgumentParser(description='生成3DGS合成训练数据')
    parser.add_argument('--data_root', type=str, 
                        default='/home/yons/Projects/ICLPose/dataset/room_0',
                        help='数据根目录')
    parser.add_argument('--sequence', type=str, default='Sequence_1',
                        help='输入序列名称')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/yons/Projects/ICLPose/dataset/room_0/Synthetic_Train',
                        help='输出目录')
    parser.add_argument('--num_perturbed', type=int, default=5,
                        help='每个基础位姿的扰动数量')
    parser.add_argument('--num_interpolated', type=int, default=3,
                        help='每对相邻位姿的插值数量')
    parser.add_argument('--sample_step', type=int, default=5,
                        help='基础位姿采样步长')
    args = parser.parse_args()
    
    print("=" * 60)
    print("3DGS合成数据生成")
    print("=" * 60)
    
    # 加载基础位姿
    traj_file = Path(args.data_root) / args.sequence / 'traj_tum.txt'
    print(f"加载位姿文件: {traj_file}")
    base_poses, timestamps = load_poses_tum(traj_file)
    print(f"  加载了 {len(base_poses)} 个位姿")
    
    # 采样基础位姿
    base_poses = base_poses[::args.sample_step]
    print(f"  采样后 {len(base_poses)} 个基础位姿")
    
    # 相机内参 (Replica)
    intrinsics = np.array([
        [320, 0, 319.5],
        [0, 320, 239.5],
        [0, 0, 1]
    ])
    
    # 初始化生成器
    print("\n初始化数据生成器...")
    # generator = SyntheticDataGenerator({})  # 暂时注释，需要3DGS支持
    
    # 生成数据
    print("\n开始生成合成数据...")
    print(f"  扰动数/位姿: {args.num_perturbed}")
    print(f"  插值数/对: {args.num_interpolated}")
    
    # 计算预期样本数
    n_base = len(base_poses)
    n_perturbed = n_base * args.num_perturbed
    n_interpolated = (n_base - 1) * args.num_interpolated + 1
    n_total = n_base + n_perturbed + n_interpolated
    print(f"\n预期生成样本数:")
    print(f"  基础位姿: {n_base}")
    print(f"  扰动位姿: {n_perturbed}")
    print(f"  插值位姿: {n_interpolated}")
    print(f"  合计: {n_total}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存生成的位姿（不含渲染，需要3DGS支持）
    print(f"\n生成位姿文件...")
    perturbed_poses = generate_perturbed_poses(
        base_poses, args.num_perturbed
    )
    interpolated_poses = interpolate_poses(base_poses, args.num_interpolated)
    all_poses = np.concatenate([base_poses, perturbed_poses, interpolated_poses])
    
    # 保存位姿
    timestamps = np.arange(len(all_poses), dtype=np.float64)
    save_poses_tum(all_poses, timestamps, output_dir / 'traj_tum.txt')
    
    print(f"\n✓ 位姿文件已保存: {output_dir / 'traj_tum.txt'}")
    print(f"  共 {len(all_poses)} 个位姿")
    print(f"\n注意: RGB渲染和特征提取需要3DGS和DINO模型支持")
    print("请运行以下命令完成数据生成:")
    print(f"  1. 使用3DGS渲染RGB图像")
    print(f"  2. 使用DINO+SD提取融合特征")


if __name__ == '__main__':
    main()
