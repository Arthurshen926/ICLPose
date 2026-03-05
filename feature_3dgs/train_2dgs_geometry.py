#!/usr/bin/env python3
"""
2DGS Geometry Reconstruction Training (RGB-only)
==================================================
独立的 2DGS 外观+几何重建训练脚本，不依赖 STDLoc。

使用 gsplat 1.4 的 rasterization_2dgs (true surfel ray-intersection)，
只训练 RGB 外观和几何参数，不训练特征嵌入（DINOv2 特征后续单独训练）。

核心特性:
  - 纯 RGB 训练: L1 + SSIM loss
  - 2DGS 正则化: distortion loss (校准后λ=0.01) + normal consistency loss
  - 单目深度监督: Pearson correlation loss (scale-shift invariant)
  - SegFormer mask 支持: 可选 (--use_mask)，默认关闭（定位时无 mask）
  - Adaptive densification: clone + split + prune + floater suppression
  - 全分辨率训练: 默认 longest_edge=0（不降采样）
  - 自动保存 checkpoint + 评估 PSNR

数据格式: COLMAP (sparse/0/cameras.bin, images.bin, points3D.bin/ply)

用法:
    CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_geometry \\
        --source_dir dataset/OldHospital \\
        --model_dir output/2dgs_models/OldHospital/v3 \\
        --images processed \\
        --iterations 30000 \\
        --longest_edge 1280 \\
        --lambda_dist 1 \\
        --lambda_normal 0.05
"""

import argparse
import json
import math
import os
import pickle
import struct
import sys
import time
from collections import namedtuple
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from torch import nn
from tqdm import tqdm

from gsplat import rasterization_2dgs, spherical_harmonics

# ── COLMAP 数据加载 ─────────────────────────────────────────────────────────

CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
ImageData = namedtuple("ImageData", ["id", "qvec", "tvec", "camera_id", "name"])

CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
}


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("I", f.read(4))[0]
            model_id = struct.unpack("i", f.read(4))[0]
            width = struct.unpack("Q", f.read(8))[0]
            height = struct.unpack("Q", f.read(8))[0]
            num_params = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f"{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = Camera(cam_id, CAMERA_MODELS[model_id].model_name, width, height, np.array(params))
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("I", f.read(4))[0]
            qvec = np.array(struct.unpack("4d", f.read(32)))
            tvec = np.array(struct.unpack("3d", f.read(24)))
            cam_id = struct.unpack("I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode()
            num_pts = struct.unpack("Q", f.read(8))[0]
            f.read(num_pts * 24)  # skip 2D points
            images[img_id] = ImageData(img_id, qvec, tvec, cam_id, name)
    return images


def read_points3d_binary(path):
    with open(path, "rb") as f:
        num = struct.unpack("Q", f.read(8))[0]
        xyzs, rgbs = [], []
        for _ in range(num):
            _ = struct.unpack("Q", f.read(8))  # point3D_id
            xyz = struct.unpack("3d", f.read(24))
            rgb = struct.unpack("3B", f.read(3))
            _ = struct.unpack("d", f.read(8))  # error
            track_len = struct.unpack("Q", f.read(8))[0]
            f.read(track_len * 8)  # skip track
            xyzs.append(xyz)
            rgbs.append(rgb)
    return np.array(xyzs, dtype=np.float32), np.array(rgbs, dtype=np.float32) / 255.0


def qvec2rotmat(qvec):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


# ── SH 工具 ─────────────────────────────────────────────────────────────────

C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


# ── SSIM ────────────────────────────────────────────────────────────────────

def _fspecial_gauss(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return g / g.sum()


def ssim(img1, img2, window_size=11, max_val=1.0):
    """Compute SSIM between two [C,H,W] images."""
    C = img1.shape[0]
    window = _fspecial_gauss(window_size, 1.5).to(img1.device)
    window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2

    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=pad, groups=C)
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1.unsqueeze(0) ** 2, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0) ** 2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d((img1 * img2).unsqueeze(0), window, padding=pad, groups=C) - mu1_mu2

    C1, C2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ── Gaussian Model 2DGS ────────────────────────────────────────────────────

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def build_rotation(r):
    """Quaternion [N,4] -> rotation matrix [N,3,3]."""
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class GaussianModel2DGS:
    """Trainable 2DGS model (RGB-only, no feature embedding)."""

    def __init__(self, sh_degree=3):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.spatial_lr_scale = 1.0
        self.percent_dense = 0.01

        # Parameters (initialized later)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)   # 2D scales only
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # Densification stats
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None

    # ── Properties ──

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def num_points(self):
        return self._xyz.shape[0]

    # ── Init from point cloud ──

    def create_from_pcd(self, xyz, colors, spatial_lr_scale):
        """Initialize from numpy arrays xyz [N,3], colors [N,3] (0-1)."""
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(xyz, dtype=torch.float32, device="cuda")
        fused_color = RGB2SH(torch.tensor(colors, dtype=torch.float32, device="cuda"))

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device="cuda")
        features[:, :3, 0] = fused_color
        # features[:, 3:, 1:] = 0.0  # already zeros

        print(f"  Initializing {fused_point_cloud.shape[0]:,} Gaussians")

        # KNN for initial scale estimation
        try:
            from simple_knn._C import distCUDA2
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 1e-7)
        except ImportError:
            print("  Warning: simple_knn not available, using fallback scale init")
            dist2 = torch.ones(fused_point_cloud.shape[0], device="cuda") * 0.001

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)  # 2D scales
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros(self.num_points, device="cuda")

    # ── Optimizer setup ──

    def training_setup(self, args):
        self.percent_dense = args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device="cuda")
        self.denom = torch.zeros((self.num_points, 1), device="cuda")

        f_rest_lr_div = getattr(args, 'f_rest_lr_divisor', 20.0)
        l = [
            {"params": [self._xyz], "lr": args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": args.feature_lr / f_rest_lr_div, "name": "f_rest"},
            {"params": [self._opacity], "lr": args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": args.rotation_lr, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # Store initial LRs for global decay schedule
        self._initial_lrs = {
            "f_dc": args.feature_lr,
            "f_rest": args.feature_lr / f_rest_lr_div,
            "opacity": args.opacity_lr,
            "scaling": args.scaling_lr,
            "rotation": args.rotation_lr,
        }
        self._lr_decay_factor = getattr(args, 'lr_decay_factor', 1.0)

        self.xyz_scheduler_args = {
            "lr_init": args.position_lr_init * self.spatial_lr_scale,
            "lr_final": args.position_lr_final * self.spatial_lr_scale,
            "lr_delay_mult": 0.01,
            "max_steps": args.iterations,
        }

    def update_learning_rate(self, iteration):
        a = self.xyz_scheduler_args
        delay_rate = a["lr_delay_mult"] + (1 - a["lr_delay_mult"]) * \
                     np.sin(0.5 * np.pi * np.clip(iteration / a["max_steps"], 0, 1))
        t = np.clip(iteration / a["max_steps"], 0, 1)
        log_lerp = np.exp(np.log(a["lr_init"]) * (1 - t) + np.log(max(a["lr_final"], 1e-10)) * t)
        lr = delay_rate * log_lerp
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = lr

        # Exponential decay for ALL other parameters (prevents overfitting)
        # Decay factor: lr_final = lr_init * decay_factor over full training
        decay_factor = getattr(self, '_lr_decay_factor', 1.0)
        if decay_factor < 1.0:
            # Compute per-iteration multiplicative decay
            decay = decay_factor ** t  # ranges from 1.0 → decay_factor
            for param_group in self.optimizer.param_groups:
                name = param_group["name"]
                if name != "xyz" and name in self._initial_lrs:
                    param_group["lr"] = self._initial_lrs[name] * decay

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ── Densification ──

    def add_densification_stats(self, grad_2d, visibility_filter, width, height):
        """Accumulate 2D gradient stats for densification."""
        grad = grad_2d.squeeze(0)  # [N, 2]
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[visibility_filter] += torch.norm(grad[visibility_filter, :2], dim=-1, keepdim=True)
        self.denom[visibility_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Clone
        self._densify_and_clone(grads, max_grad, extent)
        # Split
        self._densify_and_split(grads, max_grad, extent)

        # Prune
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_points_vs | big_points_ws
        self._prune_points(prune_mask)
        torch.cuda.empty_cache()

    def _densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected = torch.norm(grads, dim=-1) >= grad_threshold
        selected = selected & (self.get_scaling.max(dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected]
        new_dc = self._features_dc[selected]
        new_rest = self._features_rest[selected]
        new_opacity = self._opacity[selected]
        new_scaling = self._scaling[selected]
        new_rotation = self._rotation[selected]

        self._densification_postfix(new_xyz, new_dc, new_rest, new_opacity, new_scaling, new_rotation)

    def _densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init = self.num_points
        padded_grad = torch.zeros(n_init, device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected = (padded_grad >= grad_threshold) & \
                   (self.get_scaling.max(dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected].repeat(N, 1)
        stds = torch.cat([stds, torch.zeros_like(stds[:, :1])], dim=-1)  # pad 3rd dim
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected].repeat(N, 1)
        new_scaling = torch.log(self.get_scaling[selected].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected].repeat(N, 1)
        new_dc = self._features_dc[selected].repeat(N, 1, 1)
        new_rest = self._features_rest[selected].repeat(N, 1, 1)
        new_opacity = self._opacity[selected].repeat(N, 1)

        self._densification_postfix(new_xyz, new_dc, new_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat([selected, torch.zeros(N * selected.sum(), device="cuda", dtype=bool)])
        self._prune_points(prune_filter)

    def _densification_postfix(self, new_xyz, new_dc, new_rest, new_opacity, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz, "f_dc": new_dc, "f_rest": new_rest,
            "opacity": new_opacity, "scaling": new_scaling, "rotation": new_rotation,
        }
        optimizable_tensors = self._cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device="cuda")
        self.denom = torch.zeros((self.num_points, 1), device="cuda")
        self.max_radii2D = torch.zeros(self.num_points, device="cuda")

    def reset_opacity(self, reset_value=0.01):
        new_opacity = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * reset_value))
        optimizable_tensors = self._replace_tensor_in_optimizer(new_opacity, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # ── Optimizer helpers ──

    def _replace_tensor_in_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            ext = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"], torch.zeros_like(ext)], dim=0)
                stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"], torch.zeros_like(ext)], dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat([group["params"][0], ext], dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(torch.cat([group["params"][0], ext], dim=0).requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_points(self, mask):
        valid = ~mask
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][valid].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][valid].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]

    # ── Save / Load ──

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(f_dc.shape[1]):
            attrs.append(f"f_dc_{i}")
        for i in range(f_rest.shape[1]):
            attrs.append(f"f_rest_{i}")
        attrs.append("opacity")
        for i in range(scale.shape[1]):
            attrs.append(f"scale_{i}")
        for i in range(rotation.shape[1]):
            attrs.append(f"rot_{i}")

        dtype_full = [(a, "f4") for a in attrs]
        data = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scale, rotation], axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, data))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)
        print(f"  Saved {xyz.shape[0]:,} Gaussians to {path}")

    def load_ply(self, path):
        """Load Gaussian parameters from a PLY file."""
        plydata = PlyData.read(path)
        vertex = plydata['vertex']

        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        N = xyz.shape[0]

        def _sort_numeric(names):
            """Sort property names by numeric suffix (f_rest_2 before f_rest_10)."""
            return sorted(names, key=lambda x: int(x.split("_")[-1]))

        # SH features (must use numeric sort for correct ordering)
        f_dc_names = _sort_numeric([p.name for p in vertex.properties if p.name.startswith("f_dc_")])
        f_rest_names = _sort_numeric([p.name for p in vertex.properties if p.name.startswith("f_rest_")])
        f_dc = np.stack([vertex[n] for n in f_dc_names], axis=1)   # [N, 3]
        f_rest = np.stack([vertex[n] for n in f_rest_names], axis=1) if f_rest_names else np.zeros((N, 0))

        opacities = vertex['opacity'][:, np.newaxis]  # [N, 1]

        scale_names = _sort_numeric([p.name for p in vertex.properties if p.name.startswith("scale_")])
        scales = np.stack([vertex[n] for n in scale_names], axis=1)

        rot_names = _sort_numeric([p.name for p in vertex.properties if p.name.startswith("rot_")])
        rotations = np.stack([vertex[n] for n in rot_names], axis=1)

        # Set parameters
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device="cuda"))
        self._features_dc = nn.Parameter(
            torch.tensor(f_dc, dtype=torch.float32, device="cuda").reshape(N, 1, 3))
        # f_rest was saved as [N, K, 3] → transpose(1,2) → [N, 3, K] → flatten → [N, 3K]
        # So we need: [N, 3K] → reshape [N, 3, K] → transpose(1,2) → [N, K, 3]
        n_rest_per_channel = f_rest.shape[1] // 3 if f_rest.shape[1] > 0 else 0
        if n_rest_per_channel > 0:
            f_rest_tensor = torch.tensor(f_rest, dtype=torch.float32, device="cuda")
            f_rest_tensor = f_rest_tensor.reshape(N, 3, n_rest_per_channel).transpose(1, 2).contiguous()
            self._features_rest = nn.Parameter(f_rest_tensor)
        else:
            self._features_rest = nn.Parameter(torch.zeros(N, 0, 3, device="cuda"))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device="cuda"))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device="cuda"))
        self._rotation = nn.Parameter(torch.tensor(rotations, dtype=torch.float32, device="cuda"))
        self.max_radii2D = torch.zeros(N, device="cuda")
        self.spatial_lr_scale = 1.0

        print(f"  Loaded {N:,} Gaussians from {path}")


# ── Per-image Appearance Correction (Gaussian in the Wild) ───────────────

class AppearanceNetwork(nn.Module):
    """Per-image affine color correction inspired by Gaussian in the Wild.

    Learns a per-image embedding that produces an affine color transform
    (scale + bias) to absorb exposure and lighting variation between
    training images. This prevents the Gaussians from averaging different
    exposures, significantly improving per-view reconstruction quality.

    The output range is architecturally bounded:
      - scale ∈ [1 - scale_range/2, 1 + scale_range/2] (default ±20%)
      - bias  ∈ [-bias_range, +bias_range] (default ±0.05)
    This prevents the appearance network from overfitting and creating
    an excessive train/test distribution mismatch.

    During training: rendered_rgb → affine_adjust → compare with GT
    During test: use mean-embedding correction (or identity if close)
    """

    def __init__(self, n_images, embed_dim=32, scale_range=0.4, bias_range=0.05):
        super().__init__()
        self.n_images = n_images
        self.embed_dim = embed_dim
        self.scale_range = scale_range  # total range around 1.0
        self.bias_range = bias_range    # max absolute bias
        self.embedding = nn.Embedding(n_images, embed_dim)
        nn.init.zeros_(self.embedding.weight)  # start from zero → identity

        # MLP: embed → 6 affine params (3 scale + 3 bias)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )
        # Initialize last layer to zeros → identity transform at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, rgb, image_idx):
        """Apply per-image appearance correction.

        Args:
            rgb: [3, H, W] rendered image
            image_idx: int, training image index
        Returns:
            adjusted_rgb: [3, H, W]
        """
        idx = torch.tensor([image_idx], device=rgb.device)
        embed = self.embedding(idx)       # [1, embed_dim]
        params = self.mlp(embed)           # [1, 6]
        # Bounded scale: sigmoid in [0,1] → scaled to [1-r/2, 1+r/2]
        scale = torch.sigmoid(params[:, :3]) * self.scale_range + (1.0 - self.scale_range / 2)
        # Bounded bias: tanh in [-1,1] → scaled to [-bias_range, +bias_range]
        bias = torch.tanh(params[:, 3:]) * self.bias_range
        return rgb * scale[0, :, None, None] + bias[0, :, None, None]


class SpatialAppearanceNetwork(nn.Module):
    """Spatially-varying per-image appearance correction.

    Unlike AppearanceNetwork which applies a global (per-channel) affine
    transform, this model generates **per-pixel** scale/bias maps via a
    lightweight CNN decoder conditioned on per-image embeddings.

    This handles spatially non-uniform illumination changes such as:
    - Moving shadows (tree shadows shift between capture times)
    - Local specular highlights
    - Partial cloud cover / directional lighting changes

    Architecture:
        1. per-image embedding (learnable) → project to spatial feature
        2. Combine with 2D coordinate grid (positional encoding)
        3. Small convolutional decoder → 6-channel output (3 scale + 3 bias)
        4. Bounded output: scale ∈ [1-r/2, 1+r/2], bias ∈ [-b, b]

    The decoder operates at 1/8 resolution and is bilinearly upsampled
    to save memory while still capturing spatial variation.
    """

    def __init__(self, n_images, embed_dim=32, scale_range=0.4, bias_range=0.05,
                 hidden_dim=32, decoder_res=128):
        super().__init__()
        self.n_images = n_images
        self.embed_dim = embed_dim
        self.scale_range = scale_range
        self.bias_range = bias_range
        self.hidden_dim = hidden_dim
        self.decoder_res = decoder_res  # internal resolution for the decoder

        # Per-image embedding
        self.embedding = nn.Embedding(n_images, embed_dim)
        nn.init.zeros_(self.embedding.weight)

        # Project embedding → spatial feature map seed
        # Output: hidden_dim channels at decoder_res x decoder_res
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # Lightweight CNN decoder: upsample from 4x4 → decoder_res
        # 4→8→16→32→64→128 (5 upsample stages for decoder_res=128)
        self.decoder = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 8x8 → 16x16
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16x16 → 32x32
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64x64 → 128x128
            nn.ConvTranspose2d(hidden_dim // 4, 6, 4, stride=2, padding=1),
            # Output: 6 channels (3 scale param + 3 bias param)
        )

        # Initialize last conv to zeros → identity at start
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, rgb, image_idx):
        """Apply spatially-varying appearance correction.

        Args:
            rgb: [3, H, W] rendered image
            image_idx: int, training image index
        Returns:
            adjusted_rgb: [3, H, W]
        """
        _, H, W = rgb.shape
        device = rgb.device

        # Get per-image embedding
        idx = torch.tensor([image_idx], device=device)
        embed = self.embedding(idx)  # [1, embed_dim]

        # Project to spatial seed
        feat = self.embed_proj(embed)  # [1, hidden_dim * 4 * 4]
        feat = feat.view(1, self.hidden_dim, 4, 4)

        # Decode to low-res parameter maps
        param_maps = self.decoder(feat)  # [1, 6, decoder_res, decoder_res]

        # Bilinear upsample to full resolution
        param_maps = F.interpolate(param_maps, size=(H, W), mode='bilinear', align_corners=False)

        # Split into scale and bias, apply bounds
        scale_raw = param_maps[0, :3]  # [3, H, W]
        bias_raw = param_maps[0, 3:]   # [3, H, W]

        scale = torch.sigmoid(scale_raw) * self.scale_range + (1.0 - self.scale_range / 2)
        bias = torch.tanh(bias_raw) * self.bias_range

        return rgb * scale + bias


# ── WildGaussians: Per-Gaussian × Per-Image Appearance (NeurIPS 2024) ────

class WildGaussiansAppearance(nn.Module):
    """Per-Gaussian × Per-Image appearance modeling (WildGaussians, NeurIPS 2024).

    Each Gaussian has a learnable embedding (initialized with Fourier features
    from 3D position). Combined with per-image embedding and DC base color,
    an MLP predicts per-Gaussian affine color transform (scale, bias).

    Key differences from AppearanceNetwork:
      - Per-GAUSSIAN (not just per-image): same 3D point can have different
        colors in different images (crucial for multi-session data)
      - MLP operates on (image_emb, gaussian_emb, base_color) → affine
      - Output prior: 0.1 scaling ensures near-identity initialization
      - Fourier position encoding for gaussian embeddings → locality bias

    Training: Single toned render per image (override_colors with absgrad).
    All losses (L1 + DSSIM) on toned image. Gradient flows through MLP
    back to SH_DC (no detach), so SH converges to canonical appearance.
    Static masks from masks.pkl for transient object filtering.
    """

    def __init__(self, n_images, n_gaussians,
                 image_embed_dim=32, gaussian_embed_dim=24,
                 hidden_dim=128, n_hidden=2, output_scale=0.3):
        super().__init__()
        self.n_images = n_images
        self.n_gaussians = n_gaussians
        self.image_embed_dim = image_embed_dim
        self.gaussian_embed_dim = gaussian_embed_dim

        # Per-image embedding
        self.image_embedding = nn.Embedding(n_images, image_embed_dim)
        nn.init.zeros_(self.image_embedding.weight)

        # Per-Gaussian embedding (initialized later with Fourier features)
        self.gaussian_embedding = nn.Parameter(
            torch.zeros(n_gaussians, gaussian_embed_dim))

        # MLP: (image_emb + gaussian_emb + base_color) → 6 (3 scale + 3 bias)
        input_dim = image_embed_dim + gaussian_embed_dim + 3
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(hidden_dim, 6))
        self.mlp = nn.Sequential(*layers)

        # Initialize last layer to zeros → identity at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Output scaling factor (WildGaussians Appendix A.1)
        # Controls per-Gaussian color correction range: scale ∈ [1-s, 1+s], bias ∈ [-s, s]
        # 0.3 allows meaningful cross-sequence appearance correction
        self.output_scale = output_scale

    def init_gaussian_embeddings(self, xyz):
        """Initialize per-Gaussian embeddings with Fourier features from positions.

        Following WildGaussians: normalize positions to [0,1], then compute
        sin(π·p_k·2^m) and cos(π·p_k·2^m) for k=1,2,3 and m=1,...,4
        giving 3×4×2 = 24 dimensions.
        """
        with torch.no_grad():
            p = xyz.detach().clone()
            # Normalize using 97th percentile of L∞ norm
            linf = p.abs().max(dim=1).values
            q97 = torch.quantile(linf, 0.97)
            p = p / (q97 + 1e-8) * 0.5 + 0.5  # ~[0, 1]

            fourier = []
            for m in range(1, 5):  # m = 1, 2, 3, 4
                freq = math.pi * (2 ** m)
                fourier.append(torch.sin(freq * p))  # [N, 3]
                fourier.append(torch.cos(freq * p))  # [N, 3]
            fourier = torch.cat(fourier, dim=1)  # [N, 24]

            self.gaussian_embedding.data.copy_(fourier)
        print(f"    Per-Gaussian Fourier embeddings initialized from 3D positions")

    def compute_toned_colors(self, image_idx, base_colors):
        """Compute toned DC colors for a single image.

        Args:
            image_idx: int, training image index
            base_colors: [N, 3] DC colors (SH_dc * C0 + 0.5) for all Gaussians
        Returns:
            toned_colors: [N, 3] appearance-adjusted colors
        """
        N = base_colors.shape[0]
        device = base_colors.device

        # Per-image embedding: [1, image_embed_dim] → [N, image_embed_dim]
        idx_t = torch.tensor([image_idx], device=device)
        image_emb = self.image_embedding(idx_t).expand(N, -1)  # [N, 32]

        # Per-Gaussian embedding: [N, 24]
        gauss_emb = self.gaussian_embedding

        # MLP input: [N, 59]
        mlp_input = torch.cat([image_emb, gauss_emb, base_colors], dim=1)

        # Forward: [N, 59] → [N, 6]
        out = self.mlp(mlp_input)

        # Apply output scaling (identity-initialized, WildGaussians Eq. 5)
        # scale ≈ 1.0, bias ≈ 0.0 at initialization
        scale = self.output_scale * out[:, :3] + 1.0   # [N, 3]
        bias = self.output_scale * out[:, 3:]           # [N, 3]

        # Toned colors
        toned = scale * base_colors + bias
        return toned.clamp(0.0, 1.0)

    def get_base_colors(self, gaussians, cam=None):
        """Extract base colors from Gaussian SH coefficients.

        When cam is provided: evaluates FULL SH (view-dependent colors).
        When cam is None: DC-only (view-independent, for fallback/init).

        Returns [N, 3] colors in [0, 1] range.
        NOTE: No detach — gradient flows through SH back to SH_DC/SH_rest.
        """
        if cam is not None and gaussians.active_sh_degree > 0:
            # Full SH evaluation: view-dependent base colors
            viewmat = cam.get_world_view_transform()  # [4,4] W2C
            c2w = torch.inverse(viewmat)
            camera_center = c2w[:3, 3]  # [3]
            dirs = F.normalize(gaussians.get_xyz - camera_center[None], dim=-1)  # [N, 3]
            sh_features = gaussians.get_features  # [N, K, 3]
            colors = spherical_harmonics(
                gaussians.active_sh_degree, dirs, sh_features
            )  # [N, 3]
            return (colors + 0.5).clamp(0.0, 1.0)
        else:
            # DC-only fallback (degree 0 or no camera info)
            dc = gaussians._features_dc[:, 0, :]  # [N, 3]
            return (dc * C0 + 0.5).clamp(0.0, 1.0)

    def handle_densification(self, new_n_gaussians, selected_mask=None, new_xyz=None):
        """Update gaussian embeddings after densification (clone/split/prune).

        Called after Gaussian count changes. Resizes embedding and optionally
        re-initializes new Gaussians.
        """
        old_n = self.gaussian_embedding.shape[0]
        if new_n_gaussians == old_n:
            return

        if new_n_gaussians < old_n:
            # Pruning: keep only surviving Gaussians
            if selected_mask is not None:
                self.gaussian_embedding = nn.Parameter(
                    self.gaussian_embedding.data[selected_mask].contiguous())
            else:
                self.gaussian_embedding = nn.Parameter(
                    self.gaussian_embedding.data[:new_n_gaussians].contiguous())
        else:
            # Growing: pad with Fourier features of new positions
            n_new = new_n_gaussians - old_n
            if new_xyz is not None and new_xyz.shape[0] >= n_new:
                # Initialize from positions of new Gaussians
                new_embs = torch.zeros(n_new, self.gaussian_embed_dim,
                                       device=self.gaussian_embedding.device)
                # Simplified: copy mean of existing embeddings
                new_embs[:] = self.gaussian_embedding.data.mean(dim=0)
            else:
                new_embs = torch.zeros(n_new, self.gaussian_embed_dim,
                                       device=self.gaussian_embedding.device)
            self.gaussian_embedding = nn.Parameter(
                torch.cat([self.gaussian_embedding.data, new_embs], dim=0))
        self.n_gaussians = new_n_gaussians


# ── DINO Uncertainty Predictor (WildGaussians Sec. 3.3) ──────────────────

class DinoUncertaintyPredictor(nn.Module):
    """DINO-based uncertainty prediction for handling occlusions/transients.

    Uses DINOv2 ViT-B/14 cosine similarity between rendered and GT image
    features to predict per-patch uncertainty. The uncertainty is converted
    to a binary mask to exclude dynamic objects (pedestrians, cars, etc.)
    from the training loss.

    Key design (WildGaussians):
      - Uncertainty predictor: linear(GT_DINO_features) → σ (trainable)
      - Training signal: DINO cosine similarity between rendered & GT
      - Binary mask (not continuous weights) to preserve densification stats
      - DINO gradients do NOT flow back to Gaussians (detached)

    Replaces static pre-computed masks (obj_mask, sky_mask, distort_mask).
    """

    def __init__(self, dino_feature_dim=768, max_dino_size=350,
                 lambda_prior=0.5):
        super().__init__()
        self.dino_feature_dim = dino_feature_dim
        self.max_dino_size = max_dino_size
        self.lambda_prior = lambda_prior
        self.patch_size = 14  # ViT-B/14

        # Uncertainty predictor: affine transform on GT DINO features
        self.uncertainty_linear = nn.Linear(dino_feature_dim, 1)
        nn.init.zeros_(self.uncertainty_linear.weight)
        nn.init.constant_(self.uncertainty_linear.bias, 2.0)  # softplus(2)≈2.13 → initial σ≈1.46

        # DINOv2 model (loaded lazily, shared with project pipeline)
        self.dino_model = None
        self.dino_mean = None
        self.dino_std = None

        # Cache for GT DINO features {image_name: (features_cpu, patch_h, patch_w)}
        self.gt_cache = {}

    def load_dino(self, device):
        """Load DINOv2 ViT-B/14 (same model used in project feature pipeline)."""
        if self.dino_model is not None:
            return
        print("  [DINO Uncertainty] Loading DINOv2 ViT-B/14...")
        local_dir = '/home/yons/.cache/torch/hub/facebookresearch_dinov2_main'
        if os.path.isdir(local_dir):
            self.dino_model = torch.hub.load(local_dir, 'dinov2_vitb14', source='local')
        else:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dino_model.eval()
        self.dino_model.to(device)
        for p in self.dino_model.parameters():
            p.requires_grad = False
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        print("  [DINO Uncertainty] DINOv2 ViT-B/14 loaded")

    def _extract_features(self, image, device):
        """Extract patch-level DINO features from an image.

        Args:
            image: [3, H, W] tensor in [0, 1]
        Returns:
            features: [num_patches, 768], patch_h, patch_w
        """
        _, H, W = image.shape
        # Resize maintaining aspect ratio, round to patch_size
        scale = self.max_dino_size / max(H, W)
        new_H = max(self.patch_size, (int(H * scale) // self.patch_size) * self.patch_size)
        new_W = max(self.patch_size, (int(W * scale) // self.patch_size) * self.patch_size)

        img = F.interpolate(image.unsqueeze(0), size=(new_H, new_W),
                            mode='bilinear', align_corners=False)
        img = (img - self.dino_mean) / self.dino_std

        with torch.no_grad():
            out = self.dino_model.forward_features(img)
            patch_tokens = out['x_norm_patchtokens']  # [1, num_patches, 768]

        patch_h = new_H // self.patch_size
        patch_w = new_W // self.patch_size
        return patch_tokens.squeeze(0), patch_h, patch_w

    @torch.no_grad()
    def cache_gt_features(self, train_cams, load_image_fn, device):
        """Pre-extract and cache DINO features for all training images."""
        print("  [DINO Uncertainty] Pre-caching GT DINO features...")
        self.load_dino(device)
        t0 = time.time()
        for i, cam in enumerate(train_cams):
            gt_img = load_image_fn(cam).to(device)
            feats, ph, pw = self._extract_features(gt_img, device)
            self.gt_cache[cam.image_name] = (feats.cpu(), ph, pw)
            if (i + 1) % 100 == 0:
                print(f"    Cached {i+1}/{len(train_cams)} images")
        elapsed = time.time() - t0
        print(f"  [DINO Uncertainty] Cached {len(self.gt_cache)} images in {elapsed:.1f}s")

    def ensure_dino_on_device(self, device):
        """Ensure DINO model is loaded and on the correct device."""
        if self.dino_model is None:
            self.load_dino(device)
        else:
            param_device = next(self.dino_model.parameters()).device
            if param_device != device:
                self.dino_model.to(device)
                self.dino_mean = self.dino_mean.to(device)
                self.dino_std = self.dino_std.to(device)

    def compute_mask_and_loss(self, rendered_image, cam_name, target_H, target_W, device):
        """Compute binary uncertainty mask and uncertainty training loss.

        Args:
            rendered_image: [3, H, W] rendered RGB, detached from rendering graph
            cam_name: image name for GT feature lookup
            target_H, target_W: output mask spatial size
        Returns:
            mask: [1, target_H, target_W] binary mask (1=keep, 0=ignore)
            uncert_loss: scalar uncertainty training loss
        """
        if cam_name not in self.gt_cache:
            return (torch.ones(1, target_H, target_W, device=device),
                    torch.tensor(0.0, device=device))

        # Ensure DINO model is on GPU before feature extraction
        self.ensure_dino_on_device(device)

        gt_feats_cpu, patch_h, patch_w = self.gt_cache[cam_name]
        gt_feats = gt_feats_cpu.to(device)  # [num_patches, 768]

        # Extract rendered image DINO features (no grad to rendering)
        rendered_feats, _, _ = self._extract_features(rendered_image.detach(), device)

        # Cosine similarity per patch (WildGaussians Eq. 8)
        cos_sim = F.cosine_similarity(rendered_feats, gt_feats, dim=1)  # [num_patches]
        dino_loss_per_patch = torch.clamp(2.0 - 2.0 * cos_sim, min=0.0, max=1.0)

        # Uncertainty prediction from GT features
        sigma_raw = self.uncertainty_linear(gt_feats)  # [num_patches, 1]
        sigma = F.softplus(sigma_raw).squeeze(1)       # [num_patches]
        sigma = torch.clamp(sigma, min=0.1)             # minimum uncertainty

        # Uncertainty loss (WildGaussians Eq. 9): only trains uncertainty predictor
        uncert_loss = (dino_loss_per_patch.detach() / (2 * sigma ** 2)
                       + self.lambda_prior * torch.log(sigma)).mean()

        # Binary mask (WildGaussians Eq. 10)
        # Keep patches where uncertainty is low: 1/(2σ²) > 1 ⟺ σ < 1/√2
        mask_patches = ((1.0 / (2.0 * sigma ** 2)) > 1.0).float()

        # Reshape to spatial and upsample
        mask_spatial = mask_patches.view(1, 1, patch_h, patch_w)
        mask_full = F.interpolate(mask_spatial, size=(target_H, target_W),
                                  mode='nearest')
        return mask_full.squeeze(0), uncert_loss  # [1, H, W], scalar

    def unload_dino(self):
        """Free DINO model from GPU to save memory."""
        if self.dino_model is not None:
            self.dino_model.cpu()
            torch.cuda.empty_cache()


# ── Scene / Camera Data ─────────────────────────────────────────────────  ──

class CameraData:
    """Simple camera struct for training."""

    def __init__(self, uid, R, T, FovX, FovY, image, image_name, width, height):
        self.uid = uid
        self.R = R          # [3,3] numpy, transposed from COLMAP (world-to-cam R^T)
        self.T = T          # [3] numpy, translation in world-to-cam
        self.FovX = FovX
        self.FovY = FovY
        self.image = image  # [3,H,W] torch tensor 0-1, or None (lazy load)
        self.image_name = image_name
        self.width = width
        self.height = height

    def get_world_view_transform(self):
        """Return 4x4 world-to-camera matrix (OpenGL convention)."""
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.R.transpose()
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0
        return torch.tensor(Rt, dtype=torch.float32, device="cuda")


def load_scene(source_dir, images_subdir="", eval_split=True):
    """Load COLMAP scene data. Returns (train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent)."""
    sparse_dir = os.path.join(source_dir, "sparse", "0")
    cam_intrinsics = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    cam_extrinsics = read_images_binary(os.path.join(sparse_dir, "images.bin"))

    # Determine image directory
    if images_subdir:
        images_dir = os.path.join(source_dir, images_subdir)
    else:
        images_dir = os.path.join(source_dir, "images")

    # Determine test split
    test_names = set()
    list_test_path = os.path.join(sparse_dir, "list_test.txt")
    dataset_test_path = os.path.join(source_dir, "dataset_test.txt")
    if os.path.exists(list_test_path):
        with open(list_test_path) as f:
            test_names = {l.strip() for l in f if l.strip()}
    elif os.path.exists(dataset_test_path):
        with open(dataset_test_path) as f:
            for l in f:
                l = l.strip()
                if l and not l.startswith("#"):
                    test_names.add(l.split(" ")[0])

    # Load cameras
    from PIL import Image as PILImage
    all_cams = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        R = qvec2rotmat(extr.qvec).T  # COLMAP stores W2C rotation; transpose gives R for 3DGS convention
        T = extr.tvec

        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx = intr.params[0]
            FovX = focal2fov(fx, intr.width)
            FovY = focal2fov(fx, intr.height)
        elif intr.model in ("PINHOLE", "OPENCV"):
            fx, fy = intr.params[0], intr.params[1]
            FovX = focal2fov(fx, intr.width)
            FovY = focal2fov(fy, intr.height)
        else:
            raise ValueError(f"Unsupported camera model: {intr.model}")

        # Load image (lazy: store path, load on demand later)
        img_path = os.path.join(images_dir, extr.name)
        if not os.path.exists(img_path):
            # fallback: try source_dir/name directly (e.g. OldHospital/seq5/frame.png)
            img_path2 = os.path.join(source_dir, extr.name)
            if os.path.exists(img_path2):
                img_path = img_path2
            else:
                # last resort: source_dir/images/name
                img_path3 = os.path.join(source_dir, "images", extr.name)
                if os.path.exists(img_path3):
                    img_path = img_path3

        cam = CameraData(
            uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
            image=img_path,  # store path for now
            image_name=extr.name,
            width=intr.width, height=intr.height,
        )
        all_cams.append(cam)

    sys.stdout.write(f"\r  Loaded {len(all_cams)} cameras\n")

    # Split train/test
    if eval_split and test_names:
        train_cams = [c for c in all_cams if c.image_name not in test_names]
        test_cams = [c for c in all_cams if c.image_name in test_names]
        print(f"  Train: {len(train_cams)}, Test: {len(test_cams)}")
    else:
        train_cams = all_cams
        test_cams = []

    # Load point cloud
    ply_path = os.path.join(sparse_dir, "points3D.ply")
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    if os.path.exists(ply_path):
        plydata = PlyData.read(ply_path)
        vertices = plydata["vertex"]
        pcd_xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
        pcd_rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32) / 255.0
    elif os.path.exists(bin_path):
        pcd_xyz, pcd_rgb = read_points3d_binary(bin_path)
    else:
        raise FileNotFoundError(f"No point cloud found in {sparse_dir}")

    print(f"  Point cloud: {pcd_xyz.shape[0]:,} points")

    # Compute cameras extent (scene scale)
    cam_centers = []
    for c in all_cams:
        W2C = np.eye(4)
        W2C[:3, :3] = c.R.T
        W2C[:3, 3] = c.T
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3])
    cam_centers = np.array(cam_centers)
    avg_center = cam_centers.mean(axis=0)
    max_dist = np.max(np.linalg.norm(cam_centers - avg_center, axis=1))
    cameras_extent = max_dist * 1.1

    return train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent


def load_image_tensor(cam, device="cuda"):
    """Load image from path as [3,H,W] tensor on device.
    Uses cached numpy array if available (much faster than disk I/O)."""
    # Fast path: use pre-cached numpy array (avoids disk I/O)
    if hasattr(cam, '_cached_np') and cam._cached_np is not None:
        img_t = torch.from_numpy(cam._cached_np.copy()).float().permute(2, 0, 1) / 255.0
        return img_t.to(device)
    from PIL import Image as PILImage
    if isinstance(cam.image, str):
        img = PILImage.open(cam.image).convert("RGB")
        img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        return img_t.to(device)
    elif isinstance(cam.image, torch.Tensor):
        return cam.image.to(device)
    else:
        raise ValueError("CameraData.image must be path string or tensor")


# ── Rendering ───────────────────────────────────────────────────────────────

def render_2dgs(gaussians, cam, bg_color, longest_edge=0, override_colors=None):
    """Render a single view using gsplat's rasterization_2dgs.

    Args:
        override_colors: [N, D] direct per-Gaussian colors (bypass SH).
            absgrad is enabled for densification even with override_colors.

    Returns dict: render [3,H,W], rend_alpha, rend_normal, surf_normal,
                  rend_dist, viewspace_points, visibility_filter, radii, depth
    """
    means3D = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scales_2d = gaussians.get_scaling
    rotations = gaussians.get_rotation

    if override_colors is not None:
        colors = override_colors  # [N, D] direct colors
        sh_deg = None
        use_absgrad = True   # keep absgrad for densification even with override
    else:
        colors = gaussians.get_features  # [N, K, 3] SH
        sh_deg = gaussians.active_sh_degree
        use_absgrad = True

    # Pad scales to 3D for gsplat (3rd dim = 1)
    scales = torch.cat([scales_2d, torch.ones(scales_2d.shape[0], 1, device=scales_2d.device)], dim=-1)

    # Camera setup
    viewmat = cam.get_world_view_transform()  # gsplat expects [4x4] W2C row-major (no transpose!)
    width, height = cam.width, cam.height
    if longest_edge > 0:
        max_edge = max(width, height)
        if max_edge > longest_edge:
            factor = longest_edge / max_edge
            width, height = int(width * factor), int(height * factor)

    tanfovx = math.tan(cam.FovX * 0.5)
    tanfovy = math.tan(cam.FovY * 0.5)
    fx = width / (2 * tanfovx)
    fy = height / (2 * tanfovy)
    K = torch.tensor([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]], device="cuda")

    if bg_color.shape[0] == 3:
        bg4 = torch.cat([bg_color, bg_color[:1]], dim=0)
    else:
        bg4 = bg_color

    render_colors, alphas, normals, surf_normals, distort, median_depth, info = \
        rasterization_2dgs(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            packed=False,
            sh_degree=sh_deg,
            backgrounds=bg4[None],
            near_plane=0.01,
            far_plane=500,
            render_mode="RGB+ED",
            absgrad=use_absgrad,
        )

    rendered_image = render_colors[0].permute(2, 0, 1)  # [4, H, W]
    color = rendered_image[:3]
    depth = rendered_image[3:]
    radii = info["radii"].squeeze(0)

    try:
        info["gradient_2dgs"].retain_grad()
    except:
        pass

    return {
        "render": color,
        "rend_alpha": alphas,
        "rend_normal": normals,
        "surf_normal": surf_normals,
        "rend_dist": distort,
        "depth": depth,
        "viewspace_points": info["gradient_2dgs"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "width": width,
        "height": height,
    }


def render_2dgs_batch(gaussians, cams, bg_color, longest_edge=0):
    """Render multiple views in a single gsplat rasterization call.

    All cameras must render at the same resolution.
    Returns list of per-view render dicts (same format as render_2dgs).
    Significantly faster than calling render_2dgs sequentially because
    it uses a single CUDA kernel launch for all views.
    """
    B = len(cams)
    if B <= 0:
        return []
    if B == 1:
        return [render_2dgs(gaussians, cams[0], bg_color, longest_edge)]

    means3D = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scales_2d = gaussians.get_scaling
    rotations = gaussians.get_rotation
    colors = gaussians.get_features

    # Pad scales to 3D for gsplat
    scales = torch.cat([scales_2d, torch.ones(scales_2d.shape[0], 1, device=scales_2d.device)], dim=-1)

    # Compute common render resolution from first camera
    width, height = cams[0].width, cams[0].height
    if longest_edge > 0:
        max_edge = max(width, height)
        if max_edge > longest_edge:
            factor = longest_edge / max_edge
            width, height = int(width * factor), int(height * factor)

    # Build batched viewmats and Ks
    viewmats_list = []
    Ks_list = []
    for cam in cams:
        viewmat = cam.get_world_view_transform()
        tanfovx = math.tan(cam.FovX * 0.5)
        tanfovy = math.tan(cam.FovY * 0.5)
        fx = width / (2 * tanfovx)
        fy = height / (2 * tanfovy)
        K = torch.tensor([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]], device="cuda")
        viewmats_list.append(viewmat)
        Ks_list.append(K)

    viewmats_t = torch.stack(viewmats_list)  # [B, 4, 4]
    Ks_t = torch.stack(Ks_list)              # [B, 3, 3]

    if bg_color.shape[0] == 3:
        bg4 = torch.cat([bg_color, bg_color[:1]], dim=0)
    else:
        bg4 = bg_color
    backgrounds = bg4[None].expand(B, -1).contiguous()  # [B, 4]

    render_colors, alphas, normals, surf_normals, distort, median_depth, info = \
        rasterization_2dgs(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=colors,
            viewmats=viewmats_t,
            Ks=Ks_t,
            width=width,
            height=height,
            packed=False,
            sh_degree=gaussians.active_sh_degree,
            backgrounds=backgrounds,
            near_plane=0.01,
            far_plane=500,
            render_mode="RGB+ED",
            absgrad=True,
        )

    # render_colors: [B, H, W, C]  (C=4 for RGB+D)
    # alphas: [B, H, W, 1]
    # normals: [B, H, W, 3]
    # surf_normals: [B, H, W, 3]
    # distort: [B, H, W, 1]
    # info["radii"]: [B, N]
    # info["gradient_2dgs"]: [B, N, 2]

    try:
        info["gradient_2dgs"].retain_grad()
    except:
        pass

    radii_all = info["radii"]  # [B, N]

    # Unpack into per-view result dicts
    results = []
    for b in range(B):
        rendered = render_colors[b].permute(2, 0, 1)  # [C, H, W]
        results.append({
            "render": rendered[:3],
            "depth": rendered[3:],
            "rend_alpha": alphas[b:b+1] if alphas is not None else None,
            "rend_normal": normals[b:b+1] if normals is not None else None,
            "surf_normal": surf_normals[b:b+1] if surf_normals is not None else None,
            "rend_dist": distort[b:b+1] if distort is not None else None,
            "viewspace_points": info["gradient_2dgs"],  # shared [B, N, 2]
            "visibility_filter": radii_all[b] > 0,
            "radii": radii_all[b],
            "width": width,
            "height": height,
            "_batch_idx": b,
        })

    return results


# ── Training Loop ───────────────────────────────────────────────────────────

# ── Depth supervision utilities ─────────────────────────────────────────────

def pearson_depth_loss(rendered_depth, mono_depth, valid_mask=None):
    """Scale-shift invariant depth loss using Pearson correlation.

    Args:
        rendered_depth: [1, H, W] rendered depth (metric, from 2DGS)
        mono_depth: [H, W] monocular depth (normalized inverse disparity, 0-1)
        valid_mask: [H, W] optional boolean mask for valid pixels

    Returns:
        loss: 1 - pearson_correlation, clamped to [0, 2] for numerical stability
    """
    rd = rendered_depth.squeeze()
    md = mono_depth

    if valid_mask is not None:
        rd = rd[valid_mask]
        md = md[valid_mask]
    else:
        rd = rd.reshape(-1)
        md = md.reshape(-1)

    # Filter out zero-depth pixels (background/sky)
    valid = (rd > 0.01) & (md > 0.001)
    if valid.sum() < 100:
        return torch.tensor(0.0, device=rendered_depth.device)

    rd = rd[valid]
    md = md[valid]

    # Pearson correlation: invariant to scale and shift
    rd_mean = rd.mean()
    md_mean = md.mean()
    rd_c = rd - rd_mean
    md_c = md - md_mean

    rd_std = rd_c.pow(2).mean().sqrt()
    md_std = md_c.pow(2).mean().sqrt()

    # Guard: if either has near-zero variance, correlation is meaningless
    if rd_std < 1e-4 or md_std < 1e-4:
        return torch.tensor(0.0, device=rendered_depth.device)

    cov = (rd_c * md_c).mean()
    pearson = cov / (rd_std * md_std + 1e-6)  # larger epsilon for stability

    # Clamp to valid range: pearson ∈ [-1, 1], loss ∈ [0, 2]
    loss = 1.0 - pearson
    return loss.clamp(0.0, 2.0)


def load_mono_depth(cam, mono_depth_dir, target_h, target_w):
    """Load precomputed monocular depth map for a camera view.

    DPT outputs inverse disparity (larger = closer), but rendered depth is
    metric (larger = farther). We invert so both have the same direction.
    """
    depth_name = os.path.splitext(cam.image_name)[0] + ".npy"
    depth_path = os.path.join(mono_depth_dir, depth_name)
    if not os.path.exists(depth_path):
        return None

    depth = np.load(depth_path)  # [H_orig, W_orig] float32, normalized 0-1 (inverse disparity)
    # Invert: DPT has large=close, rendered depth has large=far
    depth = 1.0 - depth  # Now 0 = close, 1 = far (same direction as rendered depth)
    depth_t = torch.from_numpy(depth).float().cuda()

    # Resize to render resolution
    if depth_t.shape[0] != target_h or depth_t.shape[1] != target_w:
        depth_t = F.interpolate(
            depth_t[None, None], size=(target_h, target_w),
            mode="bilinear", align_corners=False
        ).squeeze()

    return depth_t


def train(args):
    print(f"\n{'='*60}")
    print(f"  2DGS Geometry Training (RGB-only)")
    print(f"{'='*60}")
    print(f"  Source:     {args.source_dir}")
    print(f"  Model:      {args.model_dir}")
    print(f"  Images:     {args.images or '(default)'}")
    print(f"  Iterations: {args.iterations}")
    res_str = f"longest_edge={args.longest_edge}" if args.longest_edge > 0 else "FULL (no downsample)"
    print(f"  Resolution: {res_str}")
    print(f"  Losses:     lambda_dist={args.lambda_dist}, lambda_normal={args.lambda_normal}")
    print(f"              lambda_depth={args.lambda_depth}, lambda_scale={args.lambda_scale}")
    print(f"  Mask:       {'ON' if args.use_mask else 'OFF'}")
    print(f"  Random BG:  {'ON' if getattr(args, 'random_background', False) else 'OFF'}")
    print(f"  Appearance: {'ON' if getattr(args, 'use_appearance', False) else 'OFF'}")
    print(f"  Mono depth: {args.mono_depth_dir or 'NONE'}")
    print(f"  Densify:    until_iter={args.densify_until_iter}, grad_thresh={args.densify_grad_threshold}")
    lr_decay = getattr(args, 'lr_decay_factor', 1.0)
    if lr_decay < 1.0:
        print(f"  LR decay:   {lr_decay} (all non-position params)")
    batch_size_val = getattr(args, 'batch_size', 1)
    if batch_size_val > 1:
        print(f"  Batch Size: {batch_size_val} (gradient accumulation)")
    print(f"{'='*60}\n")

    # Load scene
    print("Loading scene...")
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
        load_scene(args.source_dir, images_subdir=args.images, eval_split=True)

    # Load masks (only if --use_mask is set)
    masks = None
    if args.use_mask:
        # Search paths in priority order: explicit path, source root, images/, processed/
        candidate_paths = [
            getattr(args, 'mask_path', None),                               # --mask_path explicit
            os.path.join(args.source_dir, "masks.pkl"),                     # source_dir/masks.pkl
            os.path.join(args.source_dir, "images", "masks.pkl"),           # source_dir/images/masks.pkl
            os.path.join(args.source_dir, "processed", "masks.pkl"),        # source_dir/processed/masks.pkl
        ]
        if args.images:
            candidate_paths.insert(2, os.path.join(args.source_dir, args.images, "masks.pkl"))
        for mp in candidate_paths:
            if mp and os.path.exists(mp):
                print(f"  Loading masks from {mp}")
                masks = pickle.load(open(mp, "rb"))
                print(f"  Loaded masks for {len(masks)} images")
                # Show matched keys for first few train cams
                matched = sum(1 for c in train_cams if c.image_name in masks)
                print(f"  Matched {matched}/{len(train_cams)} training views")
                break
        if masks is None:
            print(f"  Warning: --use_mask specified but no masks.pkl found")
            print(f"  Searched: {[p for p in candidate_paths if p]}")
    else:
        print(f"  Masks disabled (use --use_mask to enable)")

    # Check mono depth directory
    mono_depth_dir = args.mono_depth_dir
    if mono_depth_dir and not os.path.isdir(mono_depth_dir):
        print(f"  Warning: mono_depth_dir={mono_depth_dir} not found, depth supervision disabled")
        mono_depth_dir = None
    if mono_depth_dir:
        # Count available depth maps
        n_depths = sum(1 for c in train_cams
                       if os.path.exists(os.path.join(mono_depth_dir, os.path.splitext(c.image_name)[0] + ".npy")))
        print(f"  Mono depth: {n_depths}/{len(train_cams)} depth maps available")

    # Initialize Gaussian model
    print("Initializing Gaussians...")
    gaussians = GaussianModel2DGS(sh_degree=args.sh_degree)
    gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)
    gaussians.training_setup(args)

    bg_color = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0],
                            dtype=torch.float32, device="cuda")

    # Per-image appearance network (Gaussian in the Wild)
    appearance_net = None
    appearance_optimizer = None
    appearance_scheduler = None
    cam_name_to_idx = {}  # image_name → index for appearance embedding

    # DINO uncertainty predictor (WildGaussians)
    dino_uncertainty = None
    dino_optimizer = None

    # WildGaussians: per-Gaussian × per-image appearance + DINO uncertainty
    use_wildgaussians = getattr(args, 'use_appearance', False) and getattr(args, 'wildgaussians', False)

    if use_wildgaussians:
        for ci, cam in enumerate(train_cams):
            cam_name_to_idx[cam.image_name] = ci

        # --- WildGaussians Appearance ---
        n_gaussians = gaussians.num_points
        wg_output_scale = getattr(args, 'wg_output_scale', 0.3)
        appearance_net = WildGaussiansAppearance(
            n_images=len(train_cams),
            n_gaussians=n_gaussians,
            image_embed_dim=32,
            gaussian_embed_dim=24,
            hidden_dim=128,
            n_hidden=2,
            output_scale=wg_output_scale,
        ).cuda()
        appearance_net.init_gaussian_embeddings(gaussians.get_xyz)

        n_params = sum(p.numel() for p in appearance_net.parameters())
        print(f"  WildGaussians appearance: {len(train_cams)} images × {n_gaussians:,} Gaussians")
        print(f"    Image embedding: 32d, Gaussian embedding: 24d (Fourier init)")
        print(f"    MLP: (32+24+3) → 128 → 128 → 6 (affine), output_scale={wg_output_scale}")
        print(f"    Total params: {n_params:,} ({n_params*4/1024/1024:.1f} MB)")

        # Separate LR for image embeddings, gaussian embeddings, and MLP
        app_lr_init = getattr(args, 'appearance_lr_init', 5e-4)
        app_lr_final = getattr(args, 'appearance_lr_final', 1e-5)
        gauss_emb_lr = getattr(args, 'gaussian_emb_lr', 5e-3)
        image_emb_lr = getattr(args, 'image_emb_lr', 1e-3)
        appearance_optimizer = torch.optim.Adam([
            {'params': [appearance_net.gaussian_embedding], 'lr': gauss_emb_lr, 'name': 'gauss_emb'},
            {'params': appearance_net.image_embedding.parameters(), 'lr': image_emb_lr, 'name': 'image_emb'},
            {'params': appearance_net.mlp.parameters(), 'lr': app_lr_init, 'name': 'app_mlp'},
        ], eps=1e-15)
        gamma = (app_lr_final / app_lr_init) ** (1.0 / max(args.iterations, 1))
        appearance_scheduler = torch.optim.lr_scheduler.ExponentialLR(appearance_optimizer, gamma=gamma)
        print(f"    LR: MLP={app_lr_init}, GaussEmb={gauss_emb_lr}, ImageEmb={image_emb_lr}")

        # --- DINO Uncertainty ---
        dino_uncertainty = DinoUncertaintyPredictor(
            dino_feature_dim=768,
            max_dino_size=getattr(args, 'dino_max_size', 350),
            lambda_prior=0.5,
        ).cuda()
        dino_optimizer = torch.optim.Adam(
            dino_uncertainty.uncertainty_linear.parameters(),
            lr=1e-3, eps=1e-15)
        print(f"    DINO uncertainty: ViT-B/14, max_size={dino_uncertainty.max_dino_size}")

    elif getattr(args, 'use_appearance', False):
        # Legacy appearance modes (AppearanceNetwork / SpatialAppearanceNetwork)
        for ci, cam in enumerate(train_cams):
            cam_name_to_idx[cam.image_name] = ci

        use_spatial = getattr(args, 'spatial_appearance', False)
        sr = getattr(args, 'appearance_scale_range', 0.4)
        br = getattr(args, 'appearance_bias_range', 0.05)

        if use_spatial:
            appearance_net = SpatialAppearanceNetwork(
                len(train_cams), embed_dim=32,
                scale_range=sr, bias_range=br,
                hidden_dim=32, decoder_res=128,
            ).cuda()
            n_params = sum(p.numel() for p in appearance_net.parameters())
            print(f"  Spatial appearance network: {len(train_cams)} embeddings (dim=32)")
            print(f"    CNN decoder: 4x4 → 128x128 → bilinear upsample to full res")
            print(f"    Total params: {n_params:,} ({n_params*4/1024/1024:.1f} MB)")
        else:
            appearance_net = AppearanceNetwork(
                len(train_cams), embed_dim=32,
                scale_range=sr, bias_range=br,
            ).cuda()
            print(f"  Appearance network: {len(train_cams)} embeddings (dim=32)")

        print(f"    Scale range: [{1.0-sr/2:.2f}, {1.0+sr/2:.2f}], Bias range: ±{br}")
        app_lr_init = getattr(args, 'appearance_lr_init', 1e-3)
        appearance_optimizer = torch.optim.Adam(appearance_net.parameters(), lr=app_lr_init, eps=1e-15)
        app_lr_final = getattr(args, 'appearance_lr_final', 1e-5)
        gamma = (app_lr_final / app_lr_init) ** (1.0 / max(args.iterations, 1))
        appearance_scheduler = torch.optim.lr_scheduler.ExponentialLR(appearance_optimizer, gamma=gamma)
        print(f"    LR: {app_lr_init} → {app_lr_final} (exponential decay)")

    # Output setup
    os.makedirs(args.model_dir, exist_ok=True)

    # Save cameras.json for visualization compatibility
    save_cameras_json(train_cams + test_cams, os.path.join(args.model_dir, "cameras.json"))

    # Pre-cache training images to avoid disk I/O during training
    print("  Pre-caching training images...")
    cache_count = 0
    for cam in train_cams:
        if isinstance(cam.image, str) and os.path.exists(cam.image):
            from PIL import Image as PILImage
            cam._cached_np = np.array(PILImage.open(cam.image).convert("RGB"), dtype=np.uint8)
            cache_count += 1
    print(f"  Cached {cache_count}/{len(train_cams)} images in CPU RAM")

    # Pre-cache mono depth maps to avoid repeated disk I/O
    mono_depth_cache = {}
    if mono_depth_dir:
        print("  Pre-caching mono depth maps...")
        for cam in train_cams:
            depth_name = os.path.splitext(cam.image_name)[0] + ".npy"
            depth_path = os.path.join(mono_depth_dir, depth_name)
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
                depth = 1.0 - depth  # Invert: DPT large=close → large=far
                mono_depth_cache[cam.image_name] = depth
        print(f"  Cached {len(mono_depth_cache)}/{len(train_cams)} mono depth maps")

    # Pre-cache DINO GT features for WildGaussians uncertainty
    if use_wildgaussians and dino_uncertainty is not None:
        dino_uncertainty.cache_gt_features(train_cams, load_image_tensor, 'cuda')
        # Unload DINO model after caching to save GPU memory during training
        # It will be re-loaded on demand for rendered image feature extraction
        dino_uncertainty.unload_dino()

    # Determine DINO update schedule (expensive, so not every iteration)
    dino_update_interval = getattr(args, 'dino_update_interval', 100)
    dino_warmup_iter = getattr(args, 'dino_warmup_iter', 2000)
    current_dino_masks = {}  # {cam_name: mask_tensor} — cached between DINO updates

    # Training loop
    viewpoint_stack = None
    ema_loss = 0.0
    batch_size = getattr(args, 'batch_size', 1)
    if batch_size > 1:
        print(f"  Batch parallel rendering: batch_size={batch_size} views per optimizer step")
    pbar = tqdm(range(1, args.iterations + 1), desc="2DGS Training")

    for iteration in pbar:
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Appearance freeze notification
        appearance_freeze_iter = getattr(args, 'appearance_freeze_iter', 0)
        if appearance_freeze_iter > 0 and iteration == appearance_freeze_iter:
            tqdm.write(f"\n  [Iter {iteration}] ★ APPEARANCE FREEZE — disabling appearance correction, "
                       f"Gaussians will now learn canonical colors directly")

        # ── Sample cameras for this step ──
        cams_batch = []
        for _ in range(batch_size):
            if not viewpoint_stack:
                viewpoint_stack = list(train_cams)
            cams_batch.append(viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1)))

        # ── Random background for training (encourages opaque Gaussians) ──
        if getattr(args, 'random_background', False):
            train_bg = torch.rand(3, device="cuda")
        else:
            train_bg = bg_color

        # ── Appearance controls ──
        appearance_freeze_iter = getattr(args, 'appearance_freeze_iter', 0)
        use_appearance_this_iter = (appearance_net is not None and
                                    (appearance_freeze_iter <= 0 or iteration < appearance_freeze_iter))

        # WildGaussians: base colors computed per-view inside loop (view-dependent SH)
        wg_active = use_wildgaussians and use_appearance_this_iter

        # ── Batch render ──
        # When WildGaussians is active: skip raw batch render (saves ~50% VRAM),
        # toned renders are computed per-view below.
        if wg_active:
            render_pkgs = [None] * batch_size  # placeholders, filled in per-view loop
            # Compute output resolution from camera + longest_edge (no probe render needed)
            _w, _h = cams_batch[0].width, cams_batch[0].height
            if args.longest_edge > 0 and max(_w, _h) > args.longest_edge:
                _f = args.longest_edge / max(_w, _h)
                _w, _h = int(_w * _f), int(_h * _f)
            rw, rh = _w, _h
        elif batch_size > 1:
            try:
                render_pkgs = render_2dgs_batch(gaussians, cams_batch, train_bg, longest_edge=args.longest_edge)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    tqdm.write(f"  [Iter {iteration}] OOM in batch render, falling back to sequential")
                    render_pkgs = [render_2dgs(gaussians, c, train_bg, longest_edge=args.longest_edge) for c in cams_batch]
                else:
                    raise
        else:
            render_pkgs = [render_2dgs(gaussians, cams_batch[0], train_bg, longest_edge=args.longest_edge)]

        if not wg_active:
            rw, rh = render_pkgs[0]["width"], render_pkgs[0]["height"]

        # ── Compute per-view losses ──
        losses = []
        batch_loss_sum = 0.0

        for _b, cam in enumerate(cams_batch):
            # ── WildGaussians: toned render is the ONLY render per view ──
            if wg_active and cam.image_name in cam_name_to_idx:
                cam_idx = cam_name_to_idx[cam.image_name]
                wg_base_colors = appearance_net.get_base_colors(gaussians, cam)  # view-dependent!
                toned_colors = appearance_net.compute_toned_colors(cam_idx, wg_base_colors)
                render_pkg = render_2dgs(gaussians, cam, train_bg,
                                         longest_edge=args.longest_edge,
                                         override_colors=toned_colors)
                render_pkgs[_b] = render_pkg
                image = render_pkg["render"]

            # ── Standard path (raw SH + optional legacy appearance) ──
            else:
                render_pkg = render_pkgs[_b]
                image = render_pkg["render"]
                if use_appearance_this_iter and cam.image_name in cam_name_to_idx:
                    image = appearance_net(image, cam_name_to_idx[cam.image_name])

            # GT image
            gt_image = load_image_tensor(cam)
            gt_image = F.interpolate(gt_image.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)

            # ── Masking ──
            mask = None        # geometry mask (depth/normal/dist)
            rgb_mask = None    # RGB loss mask (float [1, H, W])

            if masks is not None and cam.image_name in masks:
                # Legacy static masks
                obj_mask = masks[cam.image_name][0].cuda()[None]
                sky_mask = masks[cam.image_name][1].cuda()[None]
                distort_mask = masks[cam.image_name][2].cuda()[None]
                if obj_mask.shape[1] != rh or obj_mask.shape[2] != rw:
                    obj_mask = F.interpolate(obj_mask[None].float(), size=(rh, rw), mode="nearest").squeeze(0) > 0.5
                    sky_mask = F.interpolate(sky_mask[None].float(), size=(rh, rw), mode="nearest").squeeze(0) > 0.5
                    distort_mask = F.interpolate(distort_mask[None].float(), size=(rh, rw), mode="nearest").squeeze(0) > 0.5
                rgb_mask = (obj_mask & distort_mask).float()
                mask = (obj_mask & distort_mask & sky_mask).float()
                if getattr(args, 'random_background', False):
                    gt_image = gt_image * rgb_mask + train_bg[:, None, None] * (1.0 - rgb_mask)
                else:
                    image = image * rgb_mask
                    gt_image = gt_image * rgb_mask

            # ── RGB loss (all on `image` — toned for WG, corrected or raw otherwise) ──
            if rgb_mask is not None:
                n_valid = rgb_mask.sum().clamp(min=1.0)
                Ll1 = ((image - gt_image).abs() * rgb_mask).sum() / (n_valid * 3)
                ssim_val = ssim(image * rgb_mask, gt_image * rgb_mask)
            else:
                Ll1 = F.l1_loss(image, gt_image)
                ssim_val = ssim(image, gt_image)

            loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim_val)

            # 2DGS regularization losses
            lambda_normal = args.lambda_normal if iteration > args.normal_start_iter else 0.0
            lambda_dist = args.lambda_dist if iteration > args.dist_start_iter else 0.0

            if lambda_normal > 0 or lambda_dist > 0:
                rend_dist = render_pkg["rend_dist"]
                rend_normal = render_pkg["rend_normal"]
                surf_normal = render_pkg["surf_normal"]
                rend_alpha = render_pkg["rend_alpha"]

                # Normal consistency
                surf_normal_proc = surf_normal * rend_alpha.squeeze(0).detach()
                rend_normal_proc = rend_normal.squeeze(0).permute(2, 0, 1)
                if len(surf_normal_proc.shape) == 4:
                    surf_normal_proc = surf_normal_proc.squeeze(0)
                surf_normal_proc = surf_normal_proc.permute(2, 0, 1)

                normal_error = (1 - (rend_normal_proc * surf_normal_proc).sum(dim=0))[None]
                if mask is not None:
                    normal_error = normal_error * mask
                    rend_dist = rend_dist.squeeze(-1) * mask
                else:
                    rend_dist = rend_dist.squeeze(-1) if len(rend_dist.shape) > 3 else rend_dist

                if lambda_normal > 0:
                    normal_loss_val = lambda_normal * normal_error.mean()
                    loss = loss + normal_loss_val
                if lambda_dist > 0:
                    dist_loss_val = lambda_dist * rend_dist.mean()
                    loss = loss + dist_loss_val

            # Scale regularization: penalize overly large Gaussians (floater suppression)
            # Work in LOG space to avoid exp() overflow: _scaling stores log(scale),
            # threshold in log space = log(scale_reg_threshold)
            scale_loss_val = torch.tensor(0.0)
            if args.lambda_scale > 0 and iteration > args.dist_start_iter:
                log_threshold = math.log(max(args.scale_reg_threshold, 1e-6))
                max_log_scales = gaussians._scaling.detach().max(dim=1).values  # log-space, no exp
                # But we need gradient flow, so use _scaling directly:
                max_log_scales_grad = gaussians._scaling.max(dim=1).values
                excess_log = torch.clamp(max_log_scales_grad - log_threshold, min=0)
                scale_loss_val = args.lambda_scale * (excess_log ** 2).mean()
                loss = loss + scale_loss_val

            # Opacity entropy regularization: encourage binary opacity (reduce floaters)
            opacity_entropy_val = torch.tensor(0.0)
            lambda_opacity_entropy = getattr(args, 'lambda_opacity_entropy', 0.0)
            if lambda_opacity_entropy > 0 and iteration > 1000:
                o = gaussians.get_opacity.clamp(1e-6, 1 - 1e-6)
                entropy = -(o * torch.log(o) + (1 - o) * torch.log(1 - o))
                opacity_entropy_val = lambda_opacity_entropy * entropy.mean()
                loss = loss + opacity_entropy_val

            # Monocular depth supervision (Pearson correlation loss)
            depth_loss_val = torch.tensor(0.0)
            lambda_depth = args.lambda_depth if iteration > args.depth_start_iter else 0.0
            if lambda_depth > 0 and mono_depth_dir:
                rendered_depth = render_pkg["depth"]  # [1, H, W]
                # Use cached depth map (fast) or load from disk (fallback)
                if cam.image_name in mono_depth_cache:
                    depth_np = mono_depth_cache[cam.image_name]
                    mono_d = torch.from_numpy(depth_np).float().cuda()
                    if mono_d.shape[0] != rh or mono_d.shape[1] != rw:
                        mono_d = F.interpolate(
                            mono_d[None, None], size=(rh, rw),
                            mode="bilinear", align_corners=False
                        ).squeeze()
                else:
                    mono_d = load_mono_depth(cam, mono_depth_dir, rh, rw)
                if mono_d is not None:
                    # Apply mask to depth supervision: exclude dynamic objects
                    # where DPT depth is unreliable
                    depth_mask = mask.squeeze(0) if mask is not None else None
                    depth_loss_val = lambda_depth * pearson_depth_loss(
                        rendered_depth, mono_d, valid_mask=depth_mask)
                    loss = loss + depth_loss_val

            # Log loss components every 500 iters for diagnostics
            if iteration % 500 == 1 and _b == 0:
                rgb_loss_val = ((1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - ssim_val)).item()
                n_val = lambda_normal * normal_error.mean().item() if lambda_normal > 0 and 'normal_error' in dir() else 0
                d_val = lambda_dist * rend_dist.mean().item() if lambda_dist > 0 and 'rend_dist' in dir() else 0
                s_val = scale_loss_val.item() if hasattr(scale_loss_val, 'item') else 0
                dep_val = depth_loss_val.item() if hasattr(depth_loss_val, 'item') else 0
                tqdm.write(f"  [Iter {iteration}] Loss breakdown: "
                           f"RGB={rgb_loss_val:.4f} Normal={n_val:.4f} "
                           f"Dist={d_val:.4f} Scale={s_val:.6f} "
                           f"Depth={dep_val:.4f} "
                           f"Total={loss.item():.4f} N={gaussians.num_points:,}")

            losses.append(loss)
            batch_loss_sum += loss.item()

        # ── Single backward pass for all views (more efficient than per-view backward) ──
        total_loss = sum(losses) / batch_size

        # Appearance regularization (prevents train/test distribution mismatch)
        # Two modes available:
        #   1. Individual identity reg (--appearance_reg): penalizes each image's
        #      affine params deviating from identity. Keeps all corrections small.
        #   2. Mean-centering reg (--appearance_mean_reg): penalizes the POPULATION
        #      MEAN of all affine params deviating from identity, but allows individual
        #      images to deviate freely. This is better because it permits large per-
        #      image corrections (absorbing exposure variation) while keeping the raw
        #      2DGS output as the "average" appearance → good raw test PSNR.
        app_reg_weight = getattr(args, 'appearance_reg', 0.0)
        app_mean_reg = getattr(args, 'appearance_mean_reg', 0.0)
        if (app_reg_weight > 0 or app_mean_reg > 0) and appearance_net is not None and isinstance(appearance_net, AppearanceNetwork):
            # Regularization only for global AppearanceNetwork (not spatial variant)
            n_reg = min(64, appearance_net.n_images)
            reg_idx = torch.randint(0, appearance_net.n_images, (n_reg,), device='cuda')
            reg_embeds = appearance_net.embedding(reg_idx)
            reg_params = appearance_net.mlp(reg_embeds)  # [n_reg, 6]
            sr = appearance_net.scale_range
            br = appearance_net.bias_range
            reg_scale = torch.sigmoid(reg_params[:, :3]) * sr + (1.0 - sr / 2)  # target: 1.0
            reg_bias = torch.tanh(reg_params[:, 3:]) * br  # target: 0.0

            # Mode 1: Individual identity reg
            if app_reg_weight > 0:
                identity_loss = ((reg_scale - 1.0) ** 2).mean() + (reg_bias ** 2).mean()
                total_loss = total_loss + app_reg_weight * identity_loss

            # Mode 2: Mean-centering reg (allows per-image variation, constrains mean)
            if app_mean_reg > 0:
                mean_scale = reg_scale.mean(dim=0)  # [3]
                mean_bias = reg_bias.mean(dim=0)    # [3]
                mean_loss = ((mean_scale - 1.0) ** 2).sum() + (mean_bias ** 2).sum()
                total_loss = total_loss + app_mean_reg * mean_loss

        # WildGaussians regularization: L2 on image embeddings
        # Keeps embeddings small → MLP stays near identity → better test generalization
        if appearance_net is not None and isinstance(appearance_net, WildGaussiansAppearance):
            wg_emb_reg = getattr(args, 'appearance_reg', 0.0)
            if wg_emb_reg > 0:
                emb_l2 = (appearance_net.image_embedding.weight ** 2).mean()
                total_loss = total_loss + wg_emb_reg * emb_l2

        # Robust loss guard: skip backward for NaN/Inf AND extreme finite values
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss.item() > 10.0:
            tqdm.write(f"  [Iter {iteration}] WARNING: Extreme loss={total_loss.item():.4g}, skipping backward")
            gaussians.optimizer.zero_grad(set_to_none=True)
            continue

        total_loss.backward()

        # Gradient clipping to prevent optimizer state corruption from extreme updates
        if getattr(args, 'grad_clip_max_norm', 1.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                 gaussians._scaling, gaussians._rotation, gaussians._opacity],
                max_norm=args.grad_clip_max_norm
            )

        # ── Accumulate densification stats from all views in batch ──
        with torch.no_grad():
            if iteration < args.densify_until_iter:
                for _b, r_pkg in enumerate(render_pkgs):
                    vp = r_pkg["viewspace_points"]
                    grad_data = vp.grad if vp.grad is not None else vp
                    # Handle batched [B,N,2] (batch render) or single [1,N,2] (per-view/WG)
                    if grad_data.dim() == 3 and grad_data.size(0) > 1:
                        grad_data = grad_data[_b:_b+1]  # batched: extract per-view slice
                    radii = r_pkg["radii"]
                    vis = r_pkg["visibility_filter"]
                    gaussians.max_radii2D[vis] = torch.max(
                        gaussians.max_radii2D[vis], radii[vis])
                    gaussians.add_densification_stats(grad_data, vis, rw, rh)
        # ── End batch processing ──

        with torch.no_grad():
            avg_loss = batch_loss_sum / batch_size
            ema_loss = 0.4 * avg_loss + 0.6 * ema_loss
            if iteration % 10 == 0:
                pbar.set_postfix({"Loss": f"{ema_loss:.5f}", "N": f"{gaussians.num_points:,}"})

            # Save checkpoint BEFORE densification/opacity-reset
            # (avoids saving right after opacity reset when all opacities ≈ 0)
            if iteration in args.save_iterations:
                print(f"\n  [Iter {iteration}] Saving checkpoint... ({gaussians.num_points:,} Gaussians)")
                save_dir = os.path.join(args.model_dir, "point_cloud", f"iteration_{iteration}")
                gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))
                # Save appearance network alongside Gaussian checkpoint
                if appearance_net is not None:
                    if use_wildgaussians:
                        torch.save({
                            'type': 'wildgaussians',
                            'state_dict': appearance_net.state_dict(),
                            'n_images': appearance_net.n_images,
                            'n_gaussians': appearance_net.n_gaussians,
                            'image_embed_dim': appearance_net.image_embed_dim,
                            'gaussian_embed_dim': appearance_net.gaussian_embed_dim,
                        }, os.path.join(save_dir, "appearance_net.pth"))
                        if dino_uncertainty is not None:
                            torch.save({
                                'state_dict': dino_uncertainty.state_dict(),
                                'max_dino_size': dino_uncertainty.max_dino_size,
                            }, os.path.join(save_dir, "dino_uncertainty.pth"))
                    else:
                        torch.save({
                            'state_dict': appearance_net.state_dict(),
                            'n_images': appearance_net.n_images,
                            'embed_dim': appearance_net.embed_dim,
                            'scale_range': appearance_net.scale_range,
                            'bias_range': appearance_net.bias_range,
                        }, os.path.join(save_dir, "appearance_net.pth"))

            # Evaluation (also before opacity reset)
            if iteration in args.test_iterations and test_cams:
                wg_steps = getattr(args, 'wg_test_opt_steps', 0)
                evaluate(gaussians, test_cams, bg_color, args.longest_edge, iteration,
                         masks=masks, appearance_net=appearance_net,
                         use_wildgaussians=use_wildgaussians,
                         wg_test_opt_steps=wg_steps)

            # Densification
            max_gaussians = getattr(args, 'max_gaussians', 0)
            if iteration < args.densify_until_iter and (max_gaussians <= 0 or gaussians.num_points < max_gaussians):
                # Skip pruning for 500 iters after each opacity reset to let gaussians recover
                iters_since_reset = iteration % args.opacity_reset_interval
                in_cooldown = (iters_since_reset > 0 and iters_since_reset < 500) or \
                              (iteration > 0 and iters_since_reset == 0)

                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = 20 if iteration > args.opacity_reset_interval else None
                    if in_cooldown:
                        # During cooldown: only densify (clone+split), skip opacity pruning
                        gaussians.densify_and_prune(
                            args.densify_grad_threshold, 0.0,
                            cameras_extent, None,
                        )
                    else:
                        gaussians.densify_and_prune(
                            args.densify_grad_threshold, 0.005,
                            cameras_extent, size_threshold,
                        )

                if iteration % args.opacity_reset_interval == 0:
                    reset_val = getattr(args, 'opacity_reset_value', 0.01)
                    gaussians.reset_opacity(reset_value=reset_val)

                # WildGaussians: sync gaussian embeddings after densification
                if use_wildgaussians and appearance_net is not None:
                    new_n = gaussians.num_points
                    if new_n != appearance_net.n_gaussians:
                        # Reinitialize from current positions (fast, maintains locality)
                        appearance_net.n_gaussians = new_n
                        new_emb = torch.zeros(new_n, appearance_net.gaussian_embed_dim, device='cuda')
                        appearance_net.gaussian_embedding = nn.Parameter(new_emb)
                        appearance_net.init_gaussian_embeddings(gaussians.get_xyz)
                        # Update optimizer param reference
                        for pg in appearance_optimizer.param_groups:
                            if pg.get('name') == 'gauss_emb':
                                pg['params'] = [appearance_net.gaussian_embedding]
                                # Reset optimizer state for this param group
                                for p in pg['params']:
                                    if p in appearance_optimizer.state:
                                        del appearance_optimizer.state[p]
                                break

            # Post-densification: additional floater pruning every 2000 iters
            # after densification ends, with grace period after last opacity reset
            post_prune_start = args.densify_until_iter + args.opacity_reset_interval
            prune_dead_thresh = getattr(args, 'prune_dead_threshold', 0.005)
            if iteration > post_prune_start and iteration % 2000 == 0:
                with torch.no_grad():
                    max_s = gaussians.get_scaling.max(dim=1).values
                    floater_mask = (max_s > args.scale_reg_threshold * 2) & \
                                   (gaussians.get_opacity.squeeze() > 0.3)
                    dead_mask = gaussians.get_opacity.squeeze() < prune_dead_thresh
                    prune_mask = floater_mask | dead_mask
                    if prune_mask.sum() > 0:
                        n_before = gaussians.num_points
                        gaussians._prune_points(prune_mask)
                        n_pruned = n_before - gaussians.num_points
                        if n_pruned > 100:
                            tqdm.write(f"  [Iter {iteration}] Post-densify prune: {n_pruned:,} "
                                       f"(floater={floater_mask.sum().item()}, dead={dead_mask.sum().item()})")
                        # WildGaussians: sync after post-densify pruning
                        if use_wildgaussians and appearance_net is not None and n_pruned > 0:
                            new_n = gaussians.num_points
                            appearance_net.n_gaussians = new_n
                            new_emb = torch.zeros(new_n, appearance_net.gaussian_embed_dim, device='cuda')
                            appearance_net.gaussian_embedding = nn.Parameter(new_emb)
                            appearance_net.init_gaussian_embeddings(gaussians.get_xyz)
                            for pg in appearance_optimizer.param_groups:
                                if pg.get('name') == 'gauss_emb':
                                    pg['params'] = [appearance_net.gaussian_embedding]
                                    for p in pg['params']:
                                        if p in appearance_optimizer.state:
                                            del appearance_optimizer.state[p]
                                    break

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if appearance_optimizer is not None:
                appearance_optimizer.step()
                appearance_optimizer.zero_grad(set_to_none=True)
                if appearance_scheduler is not None:
                    appearance_scheduler.step()
            if dino_optimizer is not None:
                dino_optimizer.step()
                dino_optimizer.zero_grad(set_to_none=True)

    # Final save
    print(f"\n  Training complete. Saving final model...")
    save_dir = os.path.join(args.model_dir, "point_cloud", f"iteration_{args.iterations}")
    gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))
    if appearance_net is not None:
        if use_wildgaussians:
            torch.save({
                'type': 'wildgaussians',
                'state_dict': appearance_net.state_dict(),
                'n_images': appearance_net.n_images,
                'n_gaussians': appearance_net.n_gaussians,
                'image_embed_dim': appearance_net.image_embed_dim,
                'gaussian_embed_dim': appearance_net.gaussian_embed_dim,
            }, os.path.join(save_dir, "appearance_net.pth"))
            if dino_uncertainty is not None:
                torch.save({
                    'state_dict': dino_uncertainty.state_dict(),
                    'max_dino_size': dino_uncertainty.max_dino_size,
                }, os.path.join(save_dir, "dino_uncertainty.pth"))
        else:
            ckpt_data = {
                'state_dict': appearance_net.state_dict(),
                'n_images': appearance_net.n_images,
                'embed_dim': appearance_net.embed_dim,
                'scale_range': appearance_net.scale_range,
                'bias_range': appearance_net.bias_range,
                'spatial': isinstance(appearance_net, SpatialAppearanceNetwork),
            }
            if isinstance(appearance_net, SpatialAppearanceNetwork):
                ckpt_data['hidden_dim'] = appearance_net.hidden_dim
                ckpt_data['decoder_res'] = appearance_net.decoder_res
            torch.save(ckpt_data, os.path.join(save_dir, "appearance_net.pth"))

    if test_cams:
        # Final evaluation: use full test-time optimization if configured
        wg_steps_final = getattr(args, 'wg_test_opt_steps', 0)
        if wg_steps_final == 0 and use_wildgaussians:
            wg_steps_final = 100  # Full eval at end even if fast during training
        evaluate(gaussians, test_cams, bg_color, args.longest_edge, args.iterations,
                 masks=masks, appearance_net=appearance_net,
                 use_wildgaussians=use_wildgaussians,
                 wg_test_opt_steps=wg_steps_final)


def evaluate(gaussians, test_cams, bg_color, longest_edge, iteration,
             masks=None, appearance_net=None, use_wildgaussians=False,
             wg_test_opt_steps=0):
    """Evaluate PSNR on test cameras.

    Reports raw PSNR (standard) and optionally appearance-corrected PSNR.
    For WildGaussians with wg_test_opt_steps>0: per-test-image embedding optimization
    (WildGaussians paper protocol). With wg_test_opt_steps=0: mean embedding eval.
    """
    psnrs = []
    corrected_psnrs = []
    masked_psnrs = []

    # Pre-compute mean appearance correction from training embeddings
    mean_scale = None
    mean_bias = None
    if appearance_net is not None and isinstance(appearance_net, AppearanceNetwork) and not use_wildgaussians:
        with torch.no_grad():
            mean_embed = appearance_net.embedding.weight.mean(dim=0, keepdim=True)
            params = appearance_net.mlp(mean_embed)
            sr = appearance_net.scale_range
            br = appearance_net.bias_range
            mean_scale = torch.sigmoid(params[:, :3]) * sr + (1.0 - sr / 2)
            mean_bias = torch.tanh(params[:, 3:]) * br

    # WildGaussians: compute per-view toned colors for test evaluation
    wg_appearance_net = None
    if use_wildgaussians and appearance_net is not None and isinstance(appearance_net, WildGaussiansAppearance):
        wg_appearance_net = appearance_net  # used per-view below

    # Test-time embedding optimization config
    # wg_test_opt_steps=0 → fast eval with mean embedding (default during training)
    # wg_test_opt_steps>0 → per-test-image optimization (WildGaussians paper protocol)
    wg_test_opt_lr = 0.01   # learning rate for test embedding optimization

    for cam in test_cams:
        # --- Raw render & standard PSNR (no gradients needed) ---
        with torch.no_grad():
            render_pkg = render_2dgs(gaussians, cam, bg_color, longest_edge=longest_edge)
            image = render_pkg["render"].clamp(0, 1)
            rw, rh = render_pkg["width"], render_pkg["height"]

            gt_image = load_image_tensor(cam)
            gt_image = F.interpolate(gt_image.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)

            # Raw PSNR (standard metric — no appearance correction)
            mse = F.mse_loss(image, gt_image)
            if mse > 0:
                psnrs.append(-10 * math.log10(mse.item()))

            # Appearance-corrected PSNR (non-WG path)
            if mean_scale is not None:
                corrected = (image * mean_scale[0, :, None, None] + mean_bias[0, :, None, None]).clamp(0, 1)
                corr_mse = F.mse_loss(corrected, gt_image)
                if corr_mse > 0:
                    corrected_psnrs.append(-10 * math.log10(corr_mse.item()))

        # --- WildGaussians toned PSNR ---
        if wg_appearance_net is not None:
            with torch.no_grad():
                base_colors = wg_appearance_net.get_base_colors(gaussians, cam)  # view-dependent
                N_g = base_colors.shape[0]
                gauss_emb = wg_appearance_net.gaussian_embedding.detach()
                base_colors_d = base_colors.detach()
                mean_emb = wg_appearance_net.image_embedding.weight.mean(dim=0).detach()

            if wg_test_opt_steps > 0:
                # Full eval: per-test-image embedding optimization
                test_emb = mean_emb.clone()
                test_emb.requires_grad_(True)
                opt_emb = torch.optim.Adam([test_emb], lr=wg_test_opt_lr)

                for _opt_step in range(wg_test_opt_steps):
                    opt_emb.zero_grad()
                    emb_expanded = test_emb.unsqueeze(0).expand(N_g, -1)
                    mlp_in = torch.cat([emb_expanded, gauss_emb, base_colors_d], dim=1)
                    out = wg_appearance_net.mlp(mlp_in)
                    s = wg_appearance_net.output_scale * out[:, :3] + 1.0
                    b = wg_appearance_net.output_scale * out[:, 3:]
                    opt_colors = (s * base_colors_d + b).clamp(0.0, 1.0)
                    opt_pkg = render_2dgs(gaussians, cam, bg_color,
                                          longest_edge=longest_edge,
                                          override_colors=opt_colors)
                    opt_image = opt_pkg["render"]
                    opt_loss = F.l1_loss(opt_image, gt_image.detach())
                    opt_loss.backward()
                    opt_emb.step()
                final_emb = test_emb.detach()
            else:
                # Fast eval: mean training embedding
                final_emb = mean_emb

            # Toned render with chosen embedding
            with torch.no_grad():
                emb_expanded = final_emb.unsqueeze(0).expand(N_g, -1)
                mlp_in = torch.cat([emb_expanded, gauss_emb, base_colors_d], dim=1)
                out = wg_appearance_net.mlp(mlp_in)
                s = wg_appearance_net.output_scale * out[:, :3] + 1.0
                b = wg_appearance_net.output_scale * out[:, 3:]
                final_colors = (s * base_colors_d + b).clamp(0.0, 1.0)
                toned_pkg = render_2dgs(gaussians, cam, bg_color,
                                        longest_edge=longest_edge,
                                        override_colors=final_colors)
                toned_image = toned_pkg["render"].clamp(0, 1)
                toned_mse = F.mse_loss(toned_image, gt_image)
                if toned_mse > 0:
                    corrected_psnrs.append(-10 * math.log10(toned_mse.item()))

        # --- Masked PSNR ---
        with torch.no_grad():
            if masks is not None and cam.image_name in masks:
                obj_mask = masks[cam.image_name][0].cuda()[None]
                distort_mask = masks[cam.image_name][2].cuda()[None]
                rgb_mask = obj_mask & distort_mask
                if rgb_mask.shape[1] != rh or rgb_mask.shape[2] != rw:
                    rgb_mask = F.interpolate(rgb_mask[None].float(), size=(rh, rw), mode="nearest").squeeze(0) > 0.5
                n_valid = rgb_mask.sum().clamp(min=1.0)
                masked_mse = ((image - gt_image) ** 2 * rgb_mask).sum() / (n_valid * 3)
                if masked_mse > 0:
                    masked_psnrs.append(-10 * math.log10(masked_mse.item()))

    avg_psnr = sum(psnrs) / len(psnrs) if psnrs else 0
    msg = f"\n  [Iter {iteration}] Test PSNR: {avg_psnr:.2f} dB ({len(psnrs)} views)"
    if corrected_psnrs:
        avg_corr = sum(corrected_psnrs) / len(corrected_psnrs)
        label = "Toned PSNR" if use_wildgaussians else "Corrected PSNR"
        msg += f" | {label}: {avg_corr:.2f} dB"
    if masked_psnrs:
        avg_masked = sum(masked_psnrs) / len(masked_psnrs)
        msg += f" | Masked PSNR: {avg_masked:.2f} dB ({len(masked_psnrs)} views)"
    if mean_scale is not None:
        s = mean_scale.squeeze().cpu().tolist()
        b = mean_bias.squeeze().cpu().tolist()
        msg += f"\n    Appearance params: scale=[{s[0]:.4f},{s[1]:.4f},{s[2]:.4f}] bias=[{b[0]:.4f},{b[1]:.4f},{b[2]:.4f}]"
    print(msg)


def save_cameras_json(cameras, path):
    """Save camera info as JSON for visualization scripts."""
    cam_list = []
    for c in cameras:
        W2C = np.eye(4)
        W2C[:3, :3] = c.R.T
        W2C[:3, 3] = c.T
        C2W = np.linalg.inv(W2C)

        cam_list.append({
            "id": c.uid,
            "img_name": c.image_name,
            "width": c.width,
            "height": c.height,
            "fx": c.width / (2 * math.tan(c.FovX * 0.5)),
            "fy": c.height / (2 * math.tan(c.FovY * 0.5)),
            "position": C2W[:3, 3].tolist(),
            "rotation": C2W[:3, :3].tolist(),
        })

    with open(path, "w") as f:
        json.dump(cam_list, f, indent=2)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="2DGS Geometry Training (RGB-only)")

    # Data
    parser.add_argument("--source_dir", "-s", type=str, required=True, help="COLMAP dataset path")
    parser.add_argument("--model_dir", "-m", type=str, required=True, help="Output model directory")
    parser.add_argument("--images", type=str, default="", help="Subdirectory for images (e.g. 'processed')")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--use_mask", action="store_true",
                        help="Enable SegFormer mask filtering (sky/dynamic/distortion). "
                             "Default: OFF — because localization doesn't have masks.")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Explicit path to masks.pkl. If not set, searches standard locations.")
    parser.add_argument("--mono_depth_dir", type=str, default=None,
                        help="Directory containing precomputed monocular depth maps (.npy). "
                             "Use scripts/precompute_mono_depth.py to generate.")

    # Training
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of views per optimizer step (batch parallel rendering). "
                             "Uses single gsplat call for all views → better GPU utilization. "
                             "Set to 1 to disable batching. Reduce if OOM.")
    parser.add_argument("--longest_edge", type=int, default=0,
                        help="Max edge of rendering resolution (0 = full resolution, no downsampling)")

    # Learning rates
    parser.add_argument("--position_lr_init", type=float, default=0.000016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.001)
    parser.add_argument("--rotation_lr", type=float, default=0.001)

    # Losses
    parser.add_argument("--lambda_dssim", type=float, default=0.2, help="SSIM loss weight")
    parser.add_argument("--lambda_dist", type=float, default=0.01,
                        help="Distortion loss weight. CALIBRATED for gsplat: "
                             "gsplat distortion values ~0.6 (vs L1 ~0.13), "
                             "so 0.01 gives dist_loss ~= 5%% of rgb_loss. "
                             "(2DGS paper says 100 but their scale is different)")
    parser.add_argument("--lambda_normal", type=float, default=0.05,
                        help="Normal consistency loss weight")
    parser.add_argument("--normal_start_iter", type=int, default=2000,
                        help="Enable normal loss after this iteration")
    parser.add_argument("--dist_start_iter", type=int, default=3000,
                        help="Enable distortion loss after this iteration")
    parser.add_argument("--lambda_depth", type=float, default=0.1,
                        help="Monocular depth supervision weight (Pearson loss). "
                             "Requires --mono_depth_dir with precomputed depth maps.")
    parser.add_argument("--depth_start_iter", type=int, default=2000,
                        help="Enable depth supervision after this iteration")
    parser.add_argument("--lambda_scale", type=float, default=0.1,
                        help="Scale regularization weight (penalize overly large Gaussians)")
    parser.add_argument("--scale_reg_threshold", type=float, default=1.0,
                        help="Scale threshold above which regularization penalty applies")
    parser.add_argument("--lambda_opacity_entropy", type=float, default=0.0,
                        help="Opacity entropy regularization weight (encourages binary opacity)")

    # Appearance (Gaussian in the Wild)
    parser.add_argument("--use_appearance", action="store_true",
                        help="Enable per-image appearance embedding (handles exposure variation)")
    parser.add_argument("--appearance_freeze_iter", type=int, default=0,
                        help="Iteration to freeze appearance network (0=never freeze). "
                             "After this, appearance correction is disabled and Gaussians "
                             "learn canonical colors directly.")
    parser.add_argument("--grad_clip_max_norm", type=float, default=1.0,
                        help="Max norm for gradient clipping on Gaussian params. 0=disable.")
    parser.add_argument("--appearance_lr_init", type=float, default=1e-3,
                        help="Initial learning rate for appearance network")
    parser.add_argument("--appearance_lr_final", type=float, default=1e-5,
                        help="Final learning rate for appearance network (exponential decay)")
    parser.add_argument("--appearance_reg", type=float, default=0.0,
                        help="Identity-preserving regularization weight on appearance affine output")
    parser.add_argument("--appearance_mean_reg", type=float, default=0.0,
                        help="Mean-centering regularization weight: penalizes population mean of "
                             "affine params deviating from identity, allowing per-image variation")
    parser.add_argument("--appearance_scale_range", type=float, default=0.4,
                        help="Total scale range around 1.0 (e.g., 0.4 → [0.8, 1.2])")
    parser.add_argument("--appearance_bias_range", type=float, default=0.05,
                        help="Max absolute bias range (e.g., 0.05 → [-0.05, +0.05])")
    parser.add_argument("--spatial_appearance", action="store_true",
                        help="Use spatially-varying appearance (per-pixel scale/bias maps via CNN decoder). "
                             "Much more expressive than global affine — handles local shadows, specular etc.")
    parser.add_argument("--wildgaussians", action="store_true",
                        help="WildGaussians (NeurIPS 2024) appearance: per-Gaussian × per-image MLP "
                             "with DINO uncertainty masking. Requires --use_appearance. "
                             "Handles multi-session data with per-Gaussian color variation.")
    parser.add_argument("--gaussian_emb_lr", type=float, default=5e-3,
                        help="Learning rate for per-Gaussian embeddings (WildGaussians)")
    parser.add_argument("--image_emb_lr", type=float, default=1e-3,
                        help="Learning rate for per-image embeddings (WildGaussians)")
    parser.add_argument("--dino_max_size", type=int, default=350,
                        help="Max image size for DINO feature extraction (WildGaussians)")
    parser.add_argument("--dino_update_interval", type=int, default=100,
                        help="Compute DINO uncertainty mask every N iterations (expensive)")
    parser.add_argument("--dino_warmup_iter", type=int, default=2000,
                        help="Disable DINO uncertainty for first N iterations (let Gaussians initialize)")
    parser.add_argument("--wg_output_scale", type=float, default=0.3,
                        help="WildGaussians output scale: controls color correction range. "
                             "scale ∈ [1-s, 1+s], bias ∈ [-s, s]. Default 0.3.")
    parser.add_argument("--wg_test_opt_steps", type=int, default=0,
                        help="Test-time embedding optimization steps per test image (WildGaussians). "
                             "0=fast eval (mean embedding). 50-100=full eval. Slow: ~10min per eval at 100.")
    parser.add_argument("--max_gaussians", type=int, default=0,
                        help="Maximum number of Gaussians (0=unlimited). Prevents overfitting.")
    parser.add_argument("--random_background", action="store_true",
                        help="Use random background color during training (encourages opaque Gaussians, "
                             "improves sky/transparent region reconstruction)")

    # Learning rate decay
    parser.add_argument("--lr_decay_factor", type=float, default=1.0,
                        help="Global LR decay factor for non-position params. "
                             "Final LR = initial * decay_factor. E.g. 0.1 = 10x decay.")

    # Densification
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=20000,
                        help="Stop densification at this iteration")
    parser.add_argument("--densification_interval", type=int, default=100)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--percent_dense", type=float, default=0.01)
    parser.add_argument("--opacity_reset_value", type=float, default=0.01,
                        help="Opacity value after reset (sigmoid space). Higher=faster recovery.")
    parser.add_argument("--prune_dead_threshold", type=float, default=0.005,
                        help="Opacity below this are pruned post-densification")
    parser.add_argument("--f_rest_lr_divisor", type=float, default=20.0,
                        help="f_rest_lr = feature_lr / this_value. Lower=faster high-order SH")

    # Checkpoints & eval
    parser.add_argument("--save_iterations", type=int, nargs="+",
                        default=[7000, 15000, 30000])
    parser.add_argument("--test_iterations", type=int, nargs="+",
                        default=[7000, 15000, 30000])

    # Eval-only mode
    parser.add_argument("--eval_only", action="store_true",
                        help="Load a checkpoint and evaluate only (no training)")
    parser.add_argument("--checkpoint_iter", type=int, default=0,
                        help="Checkpoint iteration to load for --eval_only")

    return parser.parse_args()


def eval_only(args):
    """Load a checkpoint and run evaluation with test-time optimization."""
    ckpt_dir = os.path.join(args.model_dir, "point_cloud", f"iteration_{args.checkpoint_iter}")
    ply_path = os.path.join(ckpt_dir, "point_cloud.ply")
    app_path = os.path.join(ckpt_dir, "appearance_net.pth")

    assert os.path.exists(ply_path), f"PLY not found: {ply_path}"
    print(f"\n=== Eval-only mode: iteration {args.checkpoint_iter} ===")
    print(f"  Checkpoint: {ckpt_dir}")

    # Load scene
    train_cams, test_cams, _, _, _ = load_scene(args.source_dir, args.images)
    print(f"  Train: {len(train_cams)}, Test: {len(test_cams)}")

    # Load masks
    masks = None
    if args.use_mask:
        mask_path = getattr(args, 'mask_path', None)
        mask_candidates = [
            mask_path,
            os.path.join(args.source_dir, "masks.pkl"),
            os.path.join(args.source_dir, "masks", "masks.pkl"),
        ]
        for mp in mask_candidates:
            if mp and os.path.exists(mp):
                with open(mp, "rb") as f:
                    masks = pickle.load(f)
                print(f"  Loaded masks from {mp}")
                break

    # Background
    bg_color = torch.zeros(3, device="cuda")
    if args.white_background:
        bg_color = torch.ones(3, device="cuda")

    # Load Gaussians from PLY
    gaussians = GaussianModel2DGS(args.sh_degree)
    gaussians.load_ply(ply_path)
    print(f"  Loaded {gaussians.num_points:,} Gaussians from PLY")

    # Load appearance network
    use_wildgaussians = getattr(args, 'wildgaussians', False)
    appearance_net = None
    if os.path.exists(app_path):
        ckpt = torch.load(app_path, map_location='cuda')
        if ckpt.get('type') == 'wildgaussians' or use_wildgaussians:
            wg_output_scale = getattr(args, 'wg_output_scale', 0.3)
            appearance_net = WildGaussiansAppearance(
                n_images=ckpt['n_images'],
                n_gaussians=gaussians.num_points,
                image_embed_dim=ckpt.get('image_embed_dim', 32),
                gaussian_embed_dim=ckpt.get('gaussian_embed_dim', 24),
                output_scale=wg_output_scale,
            ).cuda()
            appearance_net.load_state_dict(ckpt['state_dict'])
            use_wildgaussians = True
            print(f"  Loaded WildGaussians appearance (output_scale={appearance_net.output_scale})")
        else:
            appearance_net = AppearanceNetwork(
                n_images=ckpt['n_images'],
                embed_dim=ckpt.get('embed_dim', 32),
                scale_range=ckpt.get('scale_range', 0.4),
                bias_range=ckpt.get('bias_range', 0.05),
            ).cuda()
            appearance_net.load_state_dict(ckpt['state_dict'])
            print(f"  Loaded AppearanceNetwork")

    # Run evaluation
    wg_steps = getattr(args, 'wg_test_opt_steps', 0)
    print(f"  WG test-time opt steps: {wg_steps}")
    print(f"  Evaluating on {len(test_cams)} test views...")

    evaluate(gaussians, test_cams, bg_color, args.longest_edge, args.checkpoint_iter,
             masks=masks, appearance_net=appearance_net,
             use_wildgaussians=use_wildgaussians,
             wg_test_opt_steps=wg_steps)


if __name__ == "__main__":
    args = parse_args()
    if getattr(args, 'eval_only', False):
        eval_only(args)
    else:
        train(args)
