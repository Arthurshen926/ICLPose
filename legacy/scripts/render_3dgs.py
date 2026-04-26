#!/usr/bin/env python3
"""
3DGS RGB渲染脚本
从给定位姿使用3D Gaussian Splatting渲染RGB图像

使用方法:
    python scripts/render_3dgs.py \
        --poses_file dataset/room_0/Synthetic_Train/traj_tum.txt \
        --output_dir dataset/room_0/Synthetic_Train/rgb \
        --ply_path dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply
"""
import os
import sys
import argparse
import math
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# 添加SplatLoc路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'reference/SplatLoc'))
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_poses_tum(tum_file):
    """加载TUM格式的位姿文件"""
    poses = []
    frame_ids = []
    
    with open(tum_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 8:
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
                
                rot = R.from_quat([qx, qy, qz, qw])
                pose = np.eye(4)
                pose[:3, :3] = rot.as_matrix()
                pose[:3, 3] = [tx, ty, tz]
                
                poses.append(pose)
                frame_ids.append(frame_num)
    
    return np.array(poses), frame_ids


class SimpleCamera:
    """简化的相机类用于渲染 - 完全兼容SplatLoc Camera"""
    
    def __init__(self, w2c, fx, fy, cx, cy, width, height, device='cuda'):
        from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
        
        self.device = device
        self.image_width = width
        self.image_height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # 计算FOV
        self.FoVx = 2 * math.atan(width / (2 * fx))
        self.FoVy = 2 * math.atan(height / (2 * fy))
        
        # W2C矩阵 (world to camera) - 确保是torch tensor
        self.W2C = torch.tensor(w2c, dtype=torch.float32, device=device)
        self.R = self.W2C[:3, :3]
        self.T = self.W2C[:3, 3]
        
        # 使用SplatLoc的getProjectionMatrix2 (完全匹配)
        self.projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, 
            cx=cx, cy=cy, fx=fx, fy=fy, 
            W=width, H=height
        ).transpose(0, 1).to(device)
    
    @property
    def world_view_transform(self):
        """获取world-to-view变换矩阵"""
        from gaussian_splatting.utils.graphics_utils import getWorld2View2
        return getWorld2View2(self.R, self.T).transpose(0, 1)
    
    @property
    def full_proj_transform(self):
        """获取完整投影变换"""
        return (self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0))).squeeze(0)
    
    @property
    def camera_center(self):
        """获取相机中心位置"""
        return self.world_view_transform.inverse()[3, :3]


class GaussianRenderer:
    """3DGS渲染器"""
    
    def __init__(self, ply_path, device='cuda'):
        self.device = device
        self.ply_path = ply_path
        
        # 加载Gaussian模型
        self._load_gaussian_model()
        
        # 背景颜色
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    def _load_gaussian_model(self):
        """加载3DGS模型"""
        from gaussian_splatting.scene.gaussian_model import GaussianModel
        
        print(f"[GaussianRenderer] 加载Gaussian模型: {self.ply_path}")
        
        # 创建最小化config供GaussianModel使用
        config = {
            "Training": {
                "primitive_reg": False
            },
            "Dataset": {
                "pcd_downsample": 64,
                "point_size": 0.01
            }
        }
        
        # 创建GaussianModel (SH degree 0, 因为PLY没有f_rest特征)
        self.gaussian_model = GaussianModel(sh_degree=0, config=config)
        self.gaussian_model.load_ply(self.ply_path)
        
        print(f"  Gaussian点数: {self.gaussian_model.get_xyz.shape[0]}")
    
    def render(self, camera):
        """
        渲染RGB图像
        
        Args:
            camera: SimpleCamera对象
            
        Returns:
            rgb: (H, W, 3) numpy数组, uint8
        """
        from gaussian_splatting.gaussian_renderer import render
        
        # 创建pipeline配置
        class PipelineParams:
            compute_cov3D_python = False
            convert_SHs_python = True  # 必须为True才能将SH转换为RGB
        
        pipe = PipelineParams()
        
        # 渲染
        with torch.no_grad():
            result = render(camera, self.gaussian_model, pipe, self.bg_color)
        
        if result is None:
            return np.zeros((camera.image_height, camera.image_width, 3), dtype=np.uint8)
        
        # 转换为numpy
        rgb = result['render'].permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        return rgb


def c2w_to_w2c(c2w):
    """将camera-to-world转换为world-to-camera"""
    w2c = np.linalg.inv(c2w)
    return w2c


def main():
    parser = argparse.ArgumentParser(description='使用3DGS渲染RGB图像')
    parser.add_argument('--poses_file', type=str, required=True,
                        help='TUM格式位姿文件')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出RGB图像目录')
    parser.add_argument('--ply_path', type=str, required=True,
                        help='3DGS PLY文件路径')
    parser.add_argument('--width', type=int, default=640,
                        help='图像宽度')
    parser.add_argument('--height', type=int, default=480,
                        help='图像高度')
    parser.add_argument('--fx', type=float, default=320.0,
                        help='相机焦距x')
    parser.add_argument('--fy', type=float, default=320.0,
                        help='相机焦距y')
    parser.add_argument('--cx', type=float, default=319.5,
                        help='相机主点x')
    parser.add_argument('--cy', type=float, default=239.5,
                        help='相机主点y')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='最大渲染帧数')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("3DGS RGB渲染")
    print("=" * 60)
    
    # 加载位姿
    print(f"\n加载位姿文件: {args.poses_file}")
    poses, frame_ids = load_poses_tum(args.poses_file)
    print(f"  加载了 {len(poses)} 个位姿")
    
    if args.max_frames:
        poses = poses[:args.max_frames]
        frame_ids = frame_ids[:args.max_frames]
        print(f"  限制渲染 {len(poses)} 帧")
    
    # 初始化渲染器
    print(f"\n初始化3DGS渲染器...")
    renderer = GaussianRenderer(args.ply_path)
    
    # 渲染
    print(f"\n开始渲染...")
    for i, (pose, frame_id) in enumerate(tqdm(zip(poses, frame_ids), 
                                               total=len(poses), desc="渲染")):
        # TUM格式的pose是camera-to-world, 需要转换为world-to-camera
        w2c = c2w_to_w2c(pose)
        
        # 创建相机
        camera = SimpleCamera(
            w2c=w2c,
            fx=args.fx, fy=args.fy,
            cx=args.cx, cy=args.cy,
            width=args.width, height=args.height
        )
        
        # 渲染
        rgb = renderer.render(camera)
        
        # 保存
        output_path = output_dir / f'frame_{i:06d}.png'
        Image.fromarray(rgb).save(str(output_path))
    
    print(f"\n✓ 完成! 渲染了 {len(poses)} 张图像")
    print(f"  输出目录: {output_dir}")


if __name__ == '__main__':
    main()
