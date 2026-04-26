"""
Hybrid Gaussian Model: 2DGS Geometry + Learnable Per-Gaussian Latent.

Extends GaussianModel2DGS with a learnable latent embedding z_i ∈ R^latent_dim.
Geometry (xyz, rotation, scaling, opacity, SH) can be trained jointly or
frozen from a pretrained reconstruction and later unfrozen for fine-tuning.
The latent participates in densification (clone/split/prune).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def build_rotation(r):
    """Build rotation matrix from quaternion [N, 4]."""
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] +
                      r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    r0, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r0 * z)
    R[:, 0, 2] = 2 * (x * z + r0 * y)
    R[:, 1, 0] = 2 * (x * y + r0 * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r0 * x)
    R[:, 2, 0] = 2 * (x * z - r0 * y)
    R[:, 2, 1] = 2 * (y * z + r0 * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def RGB2SH(rgb):
    return (rgb - 0.5) / 0.28209479177387814


class HybridGaussianModel:
    """2DGS model with learnable per-Gaussian 16d latent embedding.

    Combines:
      - Standard 2DGS geometry: xyz, rotation (quat), scaling (2D), opacity, SH colors
      - Learnable latent: z_i ∈ R^latent_dim per Gaussian

    The latent is jointly optimized and participates in densification.
    """

    def __init__(self, sh_degree: int = 3, latent_dim: int = 16):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.spatial_lr_scale = 1.0
        self.percent_dense = 0.01
        self.latent_dim = latent_dim

        # Geometry parameters
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # Learnable latent embedding
        self._latent = torch.empty(0)

        # Densification stats
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None

    # ── Properties ──

    @property
    def num_points(self):
        return self._xyz.shape[0]

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_scaling_for_render(self):
        """[N, 3] scales for gsplat (pad 2D→3D with ones)."""
        s = torch.exp(self._scaling)
        ones = torch.ones(s.shape[0], 1, device=s.device, dtype=s.dtype)
        return torch.cat([s, ones], dim=-1)

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
    def get_latent(self):
        """[N, latent_dim] raw latent vectors (no normalization)."""
        return self._latent

    @property
    def get_latent_normalized(self):
        """[N, latent_dim] L2-normalized latent for rendering."""
        return F.normalize(self._latent, p=2, dim=-1)

    # ── Initialization ──

    def create_from_pcd(self, xyz, colors, spatial_lr_scale):
        """Initialize from numpy point cloud (xyz [N,3], colors [N,3] in 0-1)."""
        self.spatial_lr_scale = spatial_lr_scale
        N = xyz.shape[0]

        fused_xyz = torch.tensor(xyz, dtype=torch.float32, device="cuda")
        fused_color = RGB2SH(torch.tensor(colors, dtype=torch.float32, device="cuda"))

        features = torch.zeros(N, 3, (self.max_sh_degree + 1) ** 2,
                               dtype=torch.float32, device="cuda")
        features[:, :3, 0] = fused_color

        # KNN for initial scale
        try:
            from simple_knn._C import distCUDA2
            dist2 = torch.clamp_min(distCUDA2(fused_xyz), 1e-7)
        except ImportError:
            dist2 = torch.ones(N, device="cuda") * 0.001

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand(N, 4, device="cuda")
        opacities = inverse_sigmoid(
            0.1 * torch.ones(N, 1, dtype=torch.float32, device="cuda")
        )

        # Latent: small random init
        latent = torch.randn(N, self.latent_dim, device="cuda") * 0.01

        self._xyz = nn.Parameter(fused_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._latent = nn.Parameter(latent.requires_grad_(True))
        self.max_radii2D = torch.zeros(N, device="cuda")

        print(f"  [HybridGaussian] Initialized {N:,} Gaussians, "
              f"latent_dim={self.latent_dim}")

    # ── Optimizer ──

    def training_setup(self, args):
        """Setup optimizer with per-parameter learning rates."""
        self.percent_dense = args.percent_dense
        self.xyz_gradient_accum = torch.zeros(self.num_points, 1, device="cuda")
        self.denom = torch.zeros(self.num_points, 1, device="cuda")

        latent_lr = getattr(args, 'latent_lr', 5e-4)
        f_rest_lr_div = getattr(args, 'f_rest_lr_divisor', 20.0)

        l = [
            {"params": [self._xyz], "lr": args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": args.feature_lr / f_rest_lr_div, "name": "f_rest"},
            {"params": [self._opacity], "lr": args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": args.rotation_lr, "name": "rotation"},
            {"params": [self._latent], "lr": latent_lr, "name": "latent"},
        ]
        # Only include params that require gradients
        l = [pg for pg in l if pg['params'][0].requires_grad]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self._initial_lrs = {
            "f_dc": args.feature_lr,
            "f_rest": args.feature_lr / f_rest_lr_div,
            "opacity": args.opacity_lr,
            "scaling": args.scaling_lr,
            "rotation": args.rotation_lr,
            "latent": latent_lr,
        }

        self.xyz_scheduler_args = {
            "lr_init": args.position_lr_init * self.spatial_lr_scale,
            "lr_final": args.position_lr_final * self.spatial_lr_scale,
            "lr_delay_mult": 0.01,
            "max_steps": args.iterations,
        }

    def set_geometry_trainable(self, trainable: bool):
        """Toggle gradients for geometry-related parameters."""
        self._xyz.requires_grad_(trainable)
        self._features_dc.requires_grad_(trainable)
        if self._features_rest.numel() > 0:
            self._features_rest.requires_grad_(trainable)
        self._scaling.requires_grad_(trainable)
        self._rotation.requires_grad_(trainable)
        self._opacity.requires_grad_(trainable)

    def update_learning_rate(self, iteration):
        """Cosine schedule for xyz, exponential decay for others."""
        a = self.xyz_scheduler_args
        delay_rate = a["lr_delay_mult"] + (1 - a["lr_delay_mult"]) * \
                     np.sin(0.5 * np.pi * np.clip(iteration / a["max_steps"], 0, 1))
        t = np.clip(iteration / a["max_steps"], 0, 1)
        log_lerp = np.exp(
            np.log(a["lr_init"]) * (1 - t) +
            np.log(max(a["lr_final"], 1e-10)) * t
        )
        lr = delay_rate * log_lerp
        for pg in self.optimizer.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = lr

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ── Densification ──

    def add_densification_stats(self, grad_2d, visibility_filter, width, height):
        grad = grad_2d.squeeze(0)
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[visibility_filter] += \
            torch.norm(grad[visibility_filter, :2], dim=-1, keepdim=True)
        self.denom[visibility_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,
                          min_gaussians=0):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self._densify_and_clone(grads, max_grad, extent)
        self._densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_vs = self.max_radii2D > max_screen_size
            big_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_vs | big_ws

        if min_gaussians > 0 and prune_mask.sum() > 0:
            n_current = self.num_points
            max_prune = max(0, n_current - min_gaussians)
            n_to_prune = prune_mask.sum().item()
            if n_to_prune > max_prune:
                if max_prune > 0:
                    ops = self.get_opacity.squeeze()
                    cand_ops = ops.clone()
                    cand_ops[~prune_mask] = float('inf')
                    _, si = cand_ops.sort()
                    new_mask = torch.zeros_like(prune_mask)
                    new_mask[si[:max_prune]] = True
                    prune_mask = new_mask
                else:
                    prune_mask = torch.zeros_like(prune_mask)

        self._prune_points(prune_mask)
        torch.cuda.empty_cache()

    def _densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected = torch.norm(grads, dim=-1) >= grad_threshold
        selected = selected & (
            self.get_scaling.max(dim=1).values <= self.percent_dense * scene_extent
        )
        new_xyz = self._xyz[selected]
        new_dc = self._features_dc[selected]
        new_rest = self._features_rest[selected]
        new_opacity = self._opacity[selected]
        new_scaling = self._scaling[selected]
        new_rotation = self._rotation[selected]
        new_latent = self._latent[selected]

        self._densification_postfix(
            new_xyz, new_dc, new_rest, new_opacity,
            new_scaling, new_rotation, new_latent,
        )

    def _densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init = self.num_points
        padded_grad = torch.zeros(n_init, device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected = (padded_grad >= grad_threshold) & \
                   (self.get_scaling.max(dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected].repeat(N, 1)
        stds = torch.cat([stds, torch.zeros_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected]).repeat(N, 1, 1)
        new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                   + self.get_xyz[selected].repeat(N, 1))
        new_scaling = torch.log(self.get_scaling[selected].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected].repeat(N, 1)
        new_dc = self._features_dc[selected].repeat(N, 1, 1)
        new_rest = self._features_rest[selected].repeat(N, 1, 1)
        new_opacity = self._opacity[selected].repeat(N, 1)
        new_latent = self._latent[selected].repeat(N, 1)

        self._densification_postfix(
            new_xyz, new_dc, new_rest, new_opacity,
            new_scaling, new_rotation, new_latent,
        )

        prune_filter = torch.cat([
            selected,
            torch.zeros(N * selected.sum(), device="cuda", dtype=bool),
        ])
        self._prune_points(prune_filter)

    def _densification_postfix(self, new_xyz, new_dc, new_rest, new_opacity,
                               new_scaling, new_rotation, new_latent):
        d = {
            "xyz": new_xyz, "f_dc": new_dc, "f_rest": new_rest,
            "opacity": new_opacity, "scaling": new_scaling,
            "rotation": new_rotation, "latent": new_latent,
        }
        opt_tensors = self._cat_tensors_to_optimizer(d)
        self._xyz = opt_tensors["xyz"]
        self._features_dc = opt_tensors["f_dc"]
        self._features_rest = opt_tensors["f_rest"]
        self._opacity = opt_tensors["opacity"]
        self._scaling = opt_tensors["scaling"]
        self._rotation = opt_tensors["rotation"]
        self._latent = opt_tensors["latent"]

        self.xyz_gradient_accum = torch.zeros(self.num_points, 1, device="cuda")
        self.denom = torch.zeros(self.num_points, 1, device="cuda")
        self.max_radii2D = torch.zeros(self.num_points, device="cuda")

    def reset_opacity(self, reset_value=0.01):
        new_op = inverse_sigmoid(
            torch.min(self.get_opacity,
                      torch.ones_like(self.get_opacity) * reset_value)
        )
        opt = self._replace_tensor_in_optimizer(new_op, "opacity")
        self._opacity = opt["opacity"]

    # ── Optimizer helpers ──

    def _replace_tensor_in_optimizer(self, tensor, name):
        out = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored = self.optimizer.state.get(group["params"][0], None)
                if stored is not None:
                    stored["exp_avg"] = torch.zeros_like(tensor)
                    stored["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored is not None:
                    self.optimizer.state[group["params"][0]] = stored
                out[group["name"]] = group["params"][0]
        return out

    def _cat_tensors_to_optimizer(self, tensors_dict):
        out = {}
        for group in self.optimizer.param_groups:
            ext = tensors_dict[group["name"]]
            stored = self.optimizer.state.get(group["params"][0], None)
            if stored is not None:
                stored["exp_avg"] = torch.cat(
                    [stored["exp_avg"], torch.zeros_like(ext)], dim=0
                )
                stored["exp_avg_sq"] = torch.cat(
                    [stored["exp_avg_sq"], torch.zeros_like(ext)], dim=0
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat([group["params"][0], ext], dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat([group["params"][0], ext], dim=0).requires_grad_(True)
                )
            out[group["name"]] = group["params"][0]
        return out

    def _prune_points(self, mask):
        valid = ~mask
        out = {}
        for group in self.optimizer.param_groups:
            stored = self.optimizer.state.get(group["params"][0], None)
            if stored is not None:
                stored["exp_avg"] = stored["exp_avg"][valid]
                stored["exp_avg_sq"] = stored["exp_avg_sq"][valid]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][valid].requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][valid].requires_grad_(True)
                )
            out[group["name"]] = group["params"][0]

        self._xyz = out["xyz"]
        self._features_dc = out["f_dc"]
        self._features_rest = out["f_rest"]
        self._opacity = out["opacity"]
        self._scaling = out["scaling"]
        self._rotation = out["rotation"]
        self._latent = out["latent"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]

    # ── Save / Load ──

    def save_ply(self, path):
        """Save to PLY with latent attributes (loc_0..loc_{latent_dim-1})."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        latent = self._latent.detach().cpu().numpy()

        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(f_dc.shape[1]):
            attrs.append(f"f_dc_{i}")
        for i in range(f_rest.shape[1]):
            attrs.append(f"f_rest_{i}")
        attrs.append("opacity")
        for i in range(scales.shape[1]):
            attrs.append(f"scale_{i}")
        for i in range(rotation.shape[1]):
            attrs.append(f"rot_{i}")
        for i in range(latent.shape[1]):
            attrs.append(f"loc_{i}")

        dtype_full = [(a, "f4") for a in attrs]
        data = np.concatenate([xyz, normals, f_dc, f_rest, opacities,
                               scales, rotation, latent], axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, data))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)
        print(f"  Saved {xyz.shape[0]:,} Gaussians to {path}")

    def load_ply(self, path, freeze_geometry=False):
        """Load from PLY. Initializes latent if not present in file."""
        plydata = PlyData.read(path)
        vertex = plydata['vertex']
        N = vertex.count

        def _sort_numeric(names):
            return sorted(names, key=lambda x: int(x.split("_")[-1]))

        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

        f_dc_names = _sort_numeric([p.name for p in vertex.properties if p.name.startswith("f_dc_")])
        f_rest_names = _sort_numeric([p.name for p in vertex.properties if p.name.startswith("f_rest_")])
        f_dc = np.stack([vertex[n] for n in f_dc_names], axis=1)
        f_rest = np.stack([vertex[n] for n in f_rest_names], axis=1) if f_rest_names else np.zeros((N, 0))

        opacities = vertex['opacity'][:, np.newaxis]

        scale_names = _sort_numeric([p.name for p in vertex.properties if p.name.startswith("scale_")])
        scales = np.stack([vertex[n] for n in scale_names], axis=1)

        rot_names = _sort_numeric([p.name for p in vertex.properties if p.name.startswith("rot_")])
        rotations = np.stack([vertex[n] for n in rot_names], axis=1)

        # Latent: load if present, else random init
        loc_names = _sort_numeric([p.name for p in vertex.properties if p.name.startswith("loc_")])
        if loc_names:
            latent = np.stack([vertex[n] for n in loc_names], axis=1)
            print(f"  Loaded {len(loc_names)}d latent from PLY")
        else:
            latent = np.random.randn(N, self.latent_dim).astype(np.float32) * 0.01
            print(f"  No latent in PLY, initialized {self.latent_dim}d random")

        device = "cuda"
        req_grad = not freeze_geometry

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device).requires_grad_(req_grad))

        n_rest_per_ch = f_rest.shape[1] // 3 if f_rest.shape[1] > 0 else 0
        self._features_dc = nn.Parameter(
            torch.tensor(f_dc, dtype=torch.float32, device=device).reshape(N, 1, 3).requires_grad_(req_grad)
        )
        if n_rest_per_ch > 0:
            fr = torch.tensor(f_rest, dtype=torch.float32, device=device)
            self._features_rest = nn.Parameter(
                fr.reshape(N, 3, n_rest_per_ch).transpose(1, 2).contiguous().requires_grad_(req_grad)
            )
        else:
            self._features_rest = nn.Parameter(
                torch.zeros(N, 0, 3, device=device).requires_grad_(False)
            )
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device).requires_grad_(req_grad))
        self._rotation = nn.Parameter(torch.tensor(rotations, dtype=torch.float32, device=device).requires_grad_(req_grad))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=device).requires_grad_(req_grad))
        self._latent = nn.Parameter(torch.tensor(latent, dtype=torch.float32, device=device).requires_grad_(True))

        self.max_radii2D = torch.zeros(N, device=device)
        self.spatial_lr_scale = 1.0

        print(f"  [HybridGaussian] Loaded {N:,} Gaussians from {path}")
        print(f"    scales: {scales.shape[1]}D ({'2DGS' if scales.shape[1] == 2 else '3DGS'})")
        print(f"    latent: {latent.shape[1]}d, geometry frozen={freeze_geometry}")
