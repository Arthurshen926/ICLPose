"""
对应关系数据集模块
用于从SplatLoc场景加载训练数据，生成2D-3D对应关系数据对
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import cv2


class CorrespondenceDataset(Dataset):
    """
    对应关系数据集类
    从SplatLoc场景加载训练数据，用于学习2D-3D对应关系
    
    数据格式：
    - RGB图像：rgb/*.png
    - 相机位姿：traj_w_c.txt (4x4矩阵，每行16个数)
    - 3D点云：从点云文件加载
    - 深度图：depth/*.png (可选)
    """
    
    def __init__(
        self,
        data_root: str,
        scene_name: str = "Sequence_1",
        image_size: Tuple[int, int] = (640, 480),
        augment: bool = True,
        max_samples: Optional[int] = None,
        use_depth: bool = False,
        gaussian_path: Optional[str] = None,
        fx: float = 320.0,
        fy: float = 320.0,
        cx: float = 319.5,
        cy: float = 239.5,
        sample_step: int = 1,
        use_initial_pose: bool = False,
        pose_noise_rot_deg: float = 5.0,
        pose_noise_trans_m: float = 0.1,
        # 🆕 视锥裁切和负样本配置
        use_init_pose_for_culling: bool = True,  # 是否使用初始位姿进行视锥裁切
        frustum_margin: float = 0.0,              # 视锥边界扩展（像素）
        negative_ratio: float = 0.0,              # 负样本比例 (0-1)
        num_pairs: int = 1024,                    # 每张图像的2D-3D对应点数量
    ):
        """
        初始化数据集
        
        参数:
            data_root: 数据根目录，如 "/home/yons/Projects/data/room_0"
            scene_name: 场景名称，如 "Sequence_1"
            image_size: 图像尺寸 (width, height)
            augment: 是否启用数据增强
            max_samples: 最大样本数量（None表示使用所有数据）
            use_depth: 是否使用深度图
            gaussian_path: Gaussian点云路径（用于特征提取）
            fx, fy, cx, cy: 相机内参
            sample_step: 采样间隔（1=使用全部帧，5=每5帧取1帧）
            use_init_pose_for_culling: 是否使用初始位姿进行视锥裁切（更符合实际场景）
            frustum_margin: 视锥边界扩展像素（正值扩大可见范围）
            negative_ratio: 负样本比例，从视锥外采样的点
        """
        super().__init__()
        
        self.data_root = data_root
        self.scene_name = scene_name
        self.scene_path = os.path.join(data_root, scene_name)
        self.image_size = image_size
        self.augment = augment
        self.use_depth = use_depth
        self.sample_step = sample_step
        
        # 初始位姿配置（用于相对位姿学习）
        self.use_initial_pose = use_initial_pose
        self.pose_noise_rot_deg = pose_noise_rot_deg
        self.pose_noise_trans_m = pose_noise_trans_m
        
        # 🆕 视锥裁切配置
        self.use_init_pose_for_culling = use_init_pose_for_culling
        self.frustum_margin = frustum_margin
        self.negative_ratio = negative_ratio
        self.num_pairs = num_pairs
        
        # 相机内参
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 加载RGB图像路径 - 支持多种命名格式
        self.rgb_dir = os.path.join(self.scene_path, "rgb")
        
        # 尝试不同的文件命名格式
        rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, "rgb_*.png")))
        self.rgb_naming = "rgb"  # 默认命名格式
        
        if len(rgb_files) == 0:
            # 尝试 frame_*.png 格式 (合成数据)
            rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, "frame_*.png")))
            self.rgb_naming = "frame"
        
        if len(rgb_files) == 0:
            # 尝试通用 *.png 格式
            rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, "*.png")))
            self.rgb_naming = "generic"
        
        if len(rgb_files) == 0:
            raise ValueError(f"未找到RGB图像: {self.rgb_dir}")
        
        # 采样策略（参考SplatLoc）
        if sample_step > 1:
            original_count = len(rgb_files)
            rgb_files = rgb_files[::sample_step]
            print(f"  [数据集] {scene_name}: 原始{original_count}帧 → 采样(step={sample_step}) → {len(rgb_files)}帧")
        
        self.rgb_files = rgb_files
        
        # 提取图像索引（从文件名中）
        self.image_indices = []
        for rgb_file in self.rgb_files:
            basename = os.path.basename(rgb_file)
            # 支持不同的命名格式
            if self.rgb_naming == "rgb":
                idx = int(basename.replace("rgb_", "").replace(".png", ""))
            elif self.rgb_naming == "frame":
                idx = int(basename.replace("frame_", "").replace(".png", ""))
            else:
                # 通用格式：尝试从文件名提取数字
                import re
                match = re.search(r'(\d+)', basename)
                idx = int(match.group(1)) if match else len(self.image_indices)
            self.image_indices.append(idx)
        
        # 限制样本数量（在采样之后）
        if max_samples is not None:
            self.rgb_files = self.rgb_files[:max_samples]
            self.image_indices = self.image_indices[:max_samples]
            print(f"  [数据集] {scene_name}: 进一步限制到前{max_samples}帧")
        
        # 加载深度图路径（如果使用）
        if self.use_depth:
            self.depth_dir = os.path.join(self.scene_path, "depth")
            if not os.path.exists(self.depth_dir):
                print(f"警告: 深度图目录不存在: {self.depth_dir}")
                self.use_depth = False
        
        # 融合特征目录 - 尝试多种可能的路径
        possible_feat_dirs = [
            os.path.join(self.scene_path, 'features_compressed/fused'),  # 原始格式
            os.path.join(self.scene_path, 'fused_feat'),                  # 合成数据格式
            os.path.join(self.scene_path, 'features_compressed'),         # 备选格式
        ]
        
        self.fused_feat_dir = None
        self.use_fused_features = False
        for feat_dir in possible_feat_dirs:
            if os.path.exists(feat_dir):
                self.fused_feat_dir = feat_dir
                self.use_fused_features = True
                break
        
        if not self.use_fused_features:
            print(f"警告: 未找到融合特征目录，尝试的路径: {possible_feat_dirs}")
            print("将在训练时动态提取特征")
        
        # 加载Gaussian点云（用于视锥裁剪）
        if gaussian_path is None:
            # 默认路径
            gaussian_path = os.path.join(data_root, 'splatloc-test/data', os.path.basename(data_root), 
                                         'point_cloud/final/point_cloud.ply')
        self.gaussian_points = self._load_gaussian_points(gaussian_path)
        
        # 加载相机位姿
        all_poses = self._load_poses()
        
        # 根据图像索引选择对应的位姿
        self.poses = np.array([all_poses[idx] for idx in self.image_indices], dtype=np.float32)
        
        # 验证数据完整性
        assert len(self.poses) == len(self.rgb_files), \
            f"位姿数量({len(self.poses)})与图像数量({len(self.rgb_files)})不匹配"
        
        # 定义图像变换
        self.transform = self._get_transforms()
        
        # 统计信息
        print(f"[CorrespondenceDataset] 已加载场景: {scene_name}")
        print(f"[CorrespondenceDataset] 图像数量: {len(self.rgb_files)}")
        print(f"[CorrespondenceDataset] 图像尺寸: {image_size}")
        print(f"[CorrespondenceDataset] 数据增强: {augment}")
        print(f"[CorrespondenceDataset] 使用深度: {use_depth}")
    
    def _load_gaussian_points(self, ply_path: str) -> np.ndarray:
        """
        从PLY文件加载Gaussian点云的3D位置
        
        参数:
            ply_path: PLY文件路径
            
        返回:
            points: [N, 3] numpy数组，世界坐标系下的3D点
        """
        from plyfile import PlyData
        
        if not os.path.exists(ply_path):
            raise ValueError(f"Gaussian点云文件不存在: {ply_path}")
        
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        
        points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
        print(f"[CorrespondenceDataset] 加载了 {len(points)} 个Gaussian点")
        
        return points.astype(np.float32)
    
    def _load_poses(self) -> np.ndarray:
        """
        加载相机位姿文件
        
        支持两种格式:
        1. traj_w_c.txt: 每行16个数字, 表示4x4变换矩阵（行优先）
        2. traj_tum.txt: TUM格式, timestamp tx ty tz qx qy qz qw
        
        返回:
            poses: [N, 4, 4] numpy数组，包含所有位姿 (c2w格式)
        """
        from scipy.spatial.transform import Rotation as R
        
        # 尝试不同的位姿文件格式
        pose_file_16 = os.path.join(self.scene_path, "traj_w_c.txt")
        pose_file_tum = os.path.join(self.scene_path, "traj_tum.txt")
        
        poses = []
        
        if os.path.exists(pose_file_16):
            # 格式1: 每行16个数字
            with open(pose_file_16, 'r') as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    if len(values) != 16:
                        continue
                    pose = np.array(values).reshape(4, 4)
                    poses.append(pose)
            print(f"[CorrespondenceDataset] 从 traj_w_c.txt 加载了 {len(poses)} 个相机位姿")
            
        elif os.path.exists(pose_file_tum):
            # 格式2: TUM格式 (timestamp tx ty tz qx qy qz qw)
            with open(pose_file_tum, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) != 8:
                        continue
                    
                    # 解析位置和四元数
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    
                    # 四元数转旋转矩阵
                    rot = R.from_quat([qx, qy, qz, qw])
                    pose = np.eye(4)
                    pose[:3, :3] = rot.as_matrix()
                    pose[:3, 3] = [tx, ty, tz]
                    poses.append(pose)
            print(f"[CorrespondenceDataset] 从 traj_tum.txt 加载了 {len(poses)} 个相机位姿")
            
        else:
            raise ValueError(f"未找到位姿文件: {pose_file_16} 或 {pose_file_tum}")
        
        return np.array(poses, dtype=np.float32)
    
    def _get_transforms(self):
        """定义图像变换"""
        transform_list = []
        
        # 基础变换
        transform_list.append(transforms.Resize(self.image_size))
        
        if self.augment:
            # 数据增强
            transform_list.extend([
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),
            ])
        
        # 转换为Tensor并归一化
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def _add_pose_noise(self, pose_gt: np.ndarray) -> np.ndarray:
        """
        给位姿添加噪声（模拟定位初始误差）
        
        参数:
            pose_gt: (4, 4) numpy数组，真值位姿
            
        返回:
            pose_noisy: (4, 4) numpy数组，带噪声的位姿
        """
        pose_noisy = pose_gt.copy()
        
        # 1. 旋转噪声：使用轴角表示
        # 生成随机旋转角度（度 -> 弧度）
        angle = np.random.randn() * (self.pose_noise_rot_deg * np.pi / 180.0)
        # 生成随机旋转轴（归一化）
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        
        # Rodrigues公式：轴角 -> 旋转矩阵
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # 应用旋转噪声：R_noisy = R_noise @ R_gt
        pose_noisy[:3, :3] = R_noise @ pose_gt[:3, :3]
        
        # 2. 平移噪声（米）
        trans_noise = np.random.randn(3) * self.pose_noise_trans_m
        pose_noisy[:3, 3] = pose_gt[:3, 3] + trans_noise
        
        return pose_noisy
    
    def _load_image(self, idx: int) -> torch.Tensor:
        """
        加载并处理RGB图像
        
        参数:
            idx: 图像索引
            
        返回:
            image: [3, H, W] tensor
        """
        img_path = self.rgb_files[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img
    
    def _load_depth(self, idx: int) -> Optional[torch.Tensor]:
        """
        加载深度图（如果可用）
        
        参数:
            idx: 图像索引
            
        返回:
            depth: [1, H, W] tensor 或 None
        """
        if not self.use_depth:
            return None
        
        # 从rgb文件名推断depth文件名
        rgb_basename = os.path.basename(self.rgb_files[idx])
        depth_name = rgb_basename.replace("rgb_", "depth_")
        depth_path = os.path.join(self.depth_dir, depth_name)
        
        if not os.path.exists(depth_path):
            return None
        
        # 加载深度图（通常是16位PNG）
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            return None
        
        # 转换数据类型并resize
        depth = depth.astype(np.float32)  # uint16 -> float32
        depth = cv2.resize(depth, self.image_size)
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        
        # 归一化（假设深度单位是mm，转换为m）
        depth = depth / 1000.0
        
        return depth
    
    def _load_fused_feature(self, idx: int) -> Optional[torch.Tensor]:
        """
        加载预提取的融合特征图
        
        支持多种格式:
        1. rgb_X_fused_768x35x46_compressed.pt (原始格式)
        2. fused_feat_XXXX.npy (合成数据格式)
        
        参数:
            idx: 图像索引
            
        返回:
            fused_feat: [256, H, W] tensor 或 None
        """
        # 如果禁用fused_features，直接返回None
        if not self.use_fused_features:
            return None
            
        if self.fused_feat_dir is None or not os.path.exists(self.fused_feat_dir):
            return None
        
        # 从rgb文件名推断融合特征文件名
        rgb_basename = os.path.basename(self.rgb_files[idx])
        img_name = rgb_basename.replace('.png', '').replace('.jpg', '')
        img_idx = self.image_indices[idx]
        
        # 尝试多种可能的文件路径
        possible_paths = [
            # 格式1: 原始 .pt 格式
            os.path.join(self.fused_feat_dir, f'{img_name}_fused_768x35x46_compressed.pt'),
            # 格式2: 合成数据 .npy 格式 (fused_feat_0000.npy)
            os.path.join(self.fused_feat_dir, f'fused_feat_{img_idx:04d}.npy'),
            # 格式3: 简化命名
            os.path.join(self.fused_feat_dir, f'{img_name}.npy'),
            os.path.join(self.fused_feat_dir, f'{img_name}.pt'),
        ]
        
        for feat_path in possible_paths:
            if not os.path.exists(feat_path):
                continue
                
            try:
                if feat_path.endswith('.npy'):
                    # 加载 .npy 格式
                    feat_data = np.load(feat_path)
                    compressed_feat = torch.from_numpy(feat_data)
                else:
                    # 加载 .pt 格式
                    fused_data = torch.load(feat_path, map_location='cpu', weights_only=True)
                    
                    # 支持两种数据格式
                    if isinstance(fused_data, dict) and 'compressed' in fused_data:
                        compressed_feat = fused_data['compressed']
                    else:
                        compressed_feat = fused_data
                
                # 处理不同的张量维度
                if compressed_feat.ndim == 4:
                    # [1, C, H, W] -> [C, H, W]
                    compressed_feat = compressed_feat.squeeze(0)
                
                # 确保是 [C, H, W] 格式
                if compressed_feat.ndim == 3:
                    if compressed_feat.shape[0] == 256:
                        feat_chw = compressed_feat  # 已经是 [C, H, W]
                    elif compressed_feat.shape[-1] == 256:
                        feat_chw = compressed_feat.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                    else:
                        print(f"警告: 未知的特征形状 {compressed_feat.shape}")
                        continue
                else:
                    print(f"警告: 特征维度不正确 {compressed_feat.shape}")
                    continue
                
                return feat_chw.float()
                
            except Exception as e:
                print(f"加载融合特征时出错 {feat_path}: {e}")
                continue
        
        # 没找到任何匹配的文件
        return None
    
    def _generate_2d_3d_pairs(
        self, 
        idx: int, 
        num_samples: int = 1024,
        pose_for_culling: np.ndarray = None,
        frustum_margin: float = 0.0,
        negative_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用视锥裁剪生成2D-3D对应关系对
        参考: SplatLoc/test.py 的 get_frusm_pts() 方法
        
        流程:
        1. 从Gaussian点云获取所有3D点
        2. 将3D点投影到指定视角的图像平面
        3. 过滤：保留深度>0且在图像范围内的点
        4. 随机采样num_samples个点
        
        参数:
            idx: 图像索引
            num_samples: 采样点数量
            pose_for_culling: 用于视锥裁切的位姿 (默认使用GT)
            frustum_margin: 视锥边界扩展 (像素), 正值扩大边界
            negative_ratio: 负样本比例 (0-1), 从视野外采样的点的比例
            
        返回:
            points_2d: [N, 2] 2D图像坐标 (u, v)
            points_3d: [N, 3] 3D世界坐标
            valid_mask: [N] 有效点掩码（True=正样本，False=负样本）
        """
        # 获取相机位姿（c2w -> w2c）
        if pose_for_culling is None:
            c2w = self.poses[idx]  # 使用GT位姿
        else:
            c2w = pose_for_culling  # 使用指定位姿（如初始位姿）
        w2c = np.linalg.inv(c2w)  # 世界到相机
        
        h, w = self.image_size[1], self.image_size[0]
        
        # 1. 获取所有Gaussian点（世界坐标系）
        all_pts = self.gaussian_points  # [N_total, 3]
        
        # 2. 转换到相机坐标系
        points_camera = (all_pts @ w2c[:3, :3].T) + w2c[:3, 3]  # [N_total, 3]
        
        # 3. 投影到图像平面
        projected_points = (self.K @ points_camera.T).T  # [N_total, 3]
        projected_points = projected_points[:, :2] / (projected_points[:, 2:3] + 1e-8)  # [N_total, 2] (u, v)
        
        # 4. 视锥裁剪：过滤在视野内的点（考虑margin）
        margin = frustum_margin
        in_frustum_mask = (points_camera[:, 2] > 0.05) & \
                          (projected_points[:, 0] >= -margin) & (projected_points[:, 0] < w + margin) & \
                          (projected_points[:, 1] >= -margin) & (projected_points[:, 1] < h + margin)
        
        # 5. 获取视野内的点（正样本）
        visible_pts_3d = all_pts[in_frustum_mask]
        visible_pts_2d = projected_points[in_frustum_mask]
        
        if len(visible_pts_3d) == 0:
            raise ValueError(f"样本 {idx}: 没有可见的Gaussian点")
        
        # 6. 计算正负样本数量
        n_positive = int(num_samples * (1 - negative_ratio))
        n_negative = num_samples - n_positive
        
        # 7. 采样正样本（视野内的点）
        n_visible = len(visible_pts_3d)
        if n_visible < n_positive:
            pos_indices = np.random.choice(n_visible, n_positive, replace=True)
        else:
            pos_indices = np.random.choice(n_visible, n_positive, replace=False)
        
        pos_pts_2d = visible_pts_2d[pos_indices]
        pos_pts_3d = visible_pts_3d[pos_indices]
        pos_mask = np.ones(n_positive, dtype=bool)
        
        # 8. 采样负样本（视野外的点）
        if n_negative > 0:
            # 视野外的点
            out_frustum_mask = ~in_frustum_mask & (points_camera[:, 2] > 0.05)  # 在相机前方但在视锥外
            out_pts_3d = all_pts[out_frustum_mask]
            out_pts_2d = projected_points[out_frustum_mask]
            
            n_out = len(out_pts_3d)
            if n_out > 0:
                if n_out < n_negative:
                    neg_indices = np.random.choice(n_out, n_negative, replace=True)
                else:
                    neg_indices = np.random.choice(n_out, n_negative, replace=False)
                neg_pts_2d = out_pts_2d[neg_indices]
                neg_pts_3d = out_pts_3d[neg_indices]
            else:
                # 没有视野外的点，用视野内的点填充
                neg_pts_2d = pos_pts_2d[:n_negative]
                neg_pts_3d = pos_pts_3d[:n_negative]
            neg_mask = np.zeros(n_negative, dtype=bool)
            
            # 合并正负样本
            points_2d = np.concatenate([pos_pts_2d, neg_pts_2d], axis=0)
            points_3d = np.concatenate([pos_pts_3d, neg_pts_3d], axis=0)
            valid_mask = np.concatenate([pos_mask, neg_mask], axis=0)
            
            # 打乱顺序
            shuffle_idx = np.random.permutation(num_samples)
            points_2d = points_2d[shuffle_idx]
            points_3d = points_3d[shuffle_idx]
            valid_mask = valid_mask[shuffle_idx]
        else:
            points_2d = pos_pts_2d
            points_3d = pos_pts_3d
            valid_mask = pos_mask
        
        # 转换为tensor
        points_2d = torch.from_numpy(points_2d.astype(np.float32))
        points_3d = torch.from_numpy(points_3d.astype(np.float32))
        valid_mask = torch.from_numpy(valid_mask)
        
        return points_2d, points_3d, valid_mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个训练样本
        
        参数:
            idx: 样本索引
            
        返回:
            sample: 包含以下键的字典
                - image: [3, H, W] RGB图像
                - pose: [4, 4] 相机位姿
                - points_2d: [N, 2] 2D图像坐标
                - points_3d: [N, 3] 3D世界坐标
                - valid_mask: [N] 有效点掩码
                - intrinsics: [3, 3] 相机内参
                - depth: [1, H, W] 深度图（可选）
        """
        # 加载图像
        image = self._load_image(idx)
        
        # 加载深度图（可选，仅用于验证）
        depth = self._load_depth(idx) if self.use_depth else None
        
        # 加载融合特征
        fused_feature = self._load_fused_feature(idx)  # [256, 480, 640] or None
        
        # 获取真值位姿
        pose_gt = self.poses[idx].copy()
        pose = torch.from_numpy(pose_gt)
        
        # 生成初始位姿（如果启用相对位姿模式）
        pose_init_np = None
        if self.use_initial_pose:
            # 训练集：添加随机噪声
            if self.augment:
                pose_init_np = self._add_pose_noise(pose_gt)
            # 验证集：添加固定小噪声或直接使用GT（可根据需要调整）
            else:
                # 验证时也添加小噪声，模拟实际定位场景
                pose_init_np = self._add_pose_noise(pose_gt)
            pose_initial = torch.from_numpy(pose_init_np)
        else:
            # 不使用初始位姿时，设为None
            pose_initial = None
        
        # 生成2D-3D对应关系
        # 🆕 支持使用初始位姿裁切 + 视锥扩展 + 负样本
        use_init_pose_for_culling = getattr(self, 'use_init_pose_for_culling', False)
        frustum_margin = getattr(self, 'frustum_margin', 0.0)
        negative_ratio = getattr(self, 'negative_ratio', 0.0)
        
        # 决定用于裁切的位姿
        if use_init_pose_for_culling and pose_init_np is not None:
            culling_pose = pose_init_np  # 使用初始位姿（带噪声）
        else:
            culling_pose = None  # 使用GT位姿
        
        points_2d, points_3d, valid_mask = self._generate_2d_3d_pairs(
            idx,
            num_samples=self.num_pairs,
            pose_for_culling=culling_pose,
            frustum_margin=frustum_margin,
            negative_ratio=negative_ratio
        )
        
        # 内参矩阵
        intrinsics = torch.from_numpy(self.K)
        
        # 构建样本字典
        sample = {
            'image': image,
            'pose': pose,
            'points_2d': points_2d,
            'points_3d': points_3d,
            'valid_mask': valid_mask,
            'fused_feature': fused_feature,  # 添加融合特征
            'intrinsics': intrinsics,
            'idx': idx,
        }
        
        # 添加初始位姿（如果启用）
        if pose_initial is not None:
            sample['initial_pose'] = pose_initial
        
        if depth is not None:
            sample['depth'] = depth
        
        return sample
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.rgb_files)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数，用于处理不同长度的点云数据
    
    参数:
        batch: 样本列表
        
    返回:
        batched: 批次数据字典
    """
    # 图像可以直接stack
    images = torch.stack([item['image'] for item in batch])
    poses = torch.stack([item['pose'] for item in batch])
    intrinsics = torch.stack([item['intrinsics'] for item in batch])
    
    # 初始位姿（如果有）
    initial_poses = None
    if 'initial_pose' in batch[0]:
        initial_poses = torch.stack([item['initial_pose'] for item in batch])
    
    # 融合特征 (可能为None)
    fused_features = []
    has_fused = all(item.get('fused_feature') is not None for item in batch)
    if has_fused:
        fused_features = torch.stack([item['fused_feature'] for item in batch])
    else:
        fused_features = None
    
    # 2D-3D点需要padding或拼接
    # 这里采用拼接方式，并记录每个样本的点数
    points_2d_list = []
    points_3d_list = []
    valid_mask_list = []
    sample_indices = []
    
    for i, item in enumerate(batch):
        n_points = item['points_2d'].shape[0]
        points_2d_list.append(item['points_2d'])
        points_3d_list.append(item['points_3d'])
        valid_mask_list.append(item['valid_mask'])
        sample_indices.append(torch.full((n_points,), i, dtype=torch.long))
    
    points_2d = torch.cat(points_2d_list, dim=0)
    points_3d = torch.cat(points_3d_list, dim=0)
    valid_mask = torch.cat(valid_mask_list, dim=0)
    sample_indices = torch.cat(sample_indices, dim=0)
    
    batched = {
        'image': images,
        'pose': poses,
        'points_2d': points_2d,
        'points_3d': points_3d,
        'valid_mask': valid_mask,
        'intrinsics': intrinsics,
        'fused_feature': fused_features,  # 添加融合特征
        'sample_indices': sample_indices,  # 记录每个点属于哪个样本
        'batch_size': len(batch),
    }
    
    # 添加初始位姿（如果有）
    if initial_poses is not None:
        batched['initial_pose'] = initial_poses
    
    # 处理深度图（如果有）
    if 'depth' in batch[0]:
        depths = torch.stack([item['depth'] for item in batch])
        batched['depth'] = depths
    
    return batched


if __name__ == '__main__':
    """测试代码"""
    # 测试数据集加载
    data_root = "/home/yons/Projects/data/room_0"
    scene_name = "Sequence_1"
    
    dataset = CorrespondenceDataset(
        data_root=data_root,
        scene_name=scene_name,
        image_size=(640, 480),
        augment=True,
        max_samples=10,
        use_depth=False,
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试单个样本
    sample = dataset[0]
    print("\n样本内容:")
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print("\n测试DataLoader:")
    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
        
        if i >= 2:  # 只测试前3个batch
            break
    
    print("\n测试完成!")
