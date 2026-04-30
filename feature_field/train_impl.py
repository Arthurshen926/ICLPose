from __future__ import annotations

"""
DCFF v5 Training — Feature Field with Online/Cached RADIO Teacher.

Key improvements over v4:
  1. Online RADIO teacher mode (no PCA bottleneck, learned projection)
  2. Cached PCA mode with existing features (backward compatible)
  3. Multi-resolution supervision (fine @ full, coarse @ half)
  4. Gradient clipping + LR warmup for training stability
  5. Larger latent_dim (32d default) for better fine feature capacity
  6. Validation every N iterations

Usage:
    # Online RADIO teacher (recommended mainline)
    CUDA_VISIBLE_DEVICES=1 python -m feature_field.train \
        --config feature_field/configs/dcff_oldhospital_v5a.yaml

    # Cached PCA fallback
    CUDA_VISIBLE_DEVICES=5 python -m feature_field.train \
        --config feature_field/configs/dcff_oldhospital_v5b.yaml
"""

import os
import sys
import math
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_field.dcff.hybrid_gaussian import HybridGaussianModel
from feature_field.dcff.hash_grid import SpatialHashGrid
from feature_field.dcff.deferred_renderer import DeferredCascadedRenderer
from feature_field.dcff.losses import DCFFLoss
from feature_field.dcff.radio_teacher import OnlineRadioTeacher, CachedFeatureTeacher
from feature_field.runtime import DepthGuidedRefiner
from feature_field.utils.checkpoint_io import safe_torch_load
from feature_field.utils.scene_colmap import (
    CameraData,
    build_da3_image_order,
    load_scene_colmap,
)
from feature_field.utils.project_config import load_feature_field_config


# Keep checkpoint readers from seeing partially written files during eval.
def _atomic_save_path(final_path, write_fn):
    final_path = Path(final_path)
    tmp_path = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
    try:
        write_fn(str(tmp_path))
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ═════════════════════════════════════════════════════════════════
# Image loading
# ═════════════════════════════════════════════════════════════════

def load_image_tensor(cam, longest_edge=960):
    """Load and resize to longest edge → [1, 3, H, W] on GPU."""
    img = Image.open(cam.image).convert('RGB')
    W, H = img.size
    scale = longest_edge / max(W, H)
    if scale < 1.0:
        img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).cuda()


def load_image_fullres(cam):
    """Load at full resolution → [1, 3, H, W] on GPU."""
    img = Image.open(cam.image).convert('RGB')
    return transforms.ToTensor()(img).unsqueeze(0).cuda()


def load_image_batch(cams, longest_edge=960, cache=None):
    if cache is not None:
        return torch.stack([cache[c.uid].float() for c in cams], dim=0)
    return torch.cat([load_image_tensor(c, longest_edge) for c in cams], dim=0)


def load_fullres_batch(cams):
    return torch.cat([load_image_fullres(c) for c in cams], dim=0)


def _extract_project_teacher_batch(
    teacher,
    teacher_images: torch.Tensor,
    teacher_mode: str,
    micro_batch: int = 0,
    return_compression_loss: bool = True,
):
    """Run online teacher extraction/projection in micro-batches.

    The rendered map still trains with the full batch.  Only the frozen RADIO
    feature extraction is split to keep peak VRAM below 24GB on 4090s.
    """
    batch = teacher_images.shape[0]
    micro_batch = int(micro_batch or batch)
    micro_batch = max(1, min(micro_batch, batch))
    fine_parts = []
    coarse_parts = []
    loss_sums = {}

    for start in range(0, batch, micro_batch):
        end = min(start + micro_batch, batch)
        chunk = teacher_images[start:end]
        fine_raw, coarse_raw = teacher.extract_raw(chunk)
        if teacher_mode == 'online_bottleneck' and return_compression_loss:
            fine_proj, coarse_proj, chunk_losses = teacher.project(
                fine_raw, coarse_raw, return_loss=True,
            )
            weight = (end - start) / batch
            for name, value in chunk_losses.items():
                loss_sums[name] = loss_sums.get(name, value.new_tensor(0.0)) + value * weight
        elif teacher_mode == 'online_bottleneck':
            with torch.no_grad():
                fine_proj, coarse_proj = teacher.project(fine_raw, coarse_raw)
        else:
            fine_proj, coarse_proj = teacher.project(fine_raw, coarse_raw)

        fine_parts.append(fine_proj)
        coarse_parts.append(coarse_proj)
        del fine_raw, coarse_raw

    return torch.cat(fine_parts, dim=0), torch.cat(coarse_parts, dim=0), loss_sums


def _cache_resized_images(cams, longest_edge, dtype=torch.float16, show_progress=True):
    cache = {}
    total = len(cams)
    for idx, cam in enumerate(cams):
        cache[cam.uid] = load_image_tensor(cam, longest_edge).squeeze(0).to(dtype=dtype)
        if show_progress and (idx + 1) % 200 == 0:
            print(f"  cached {idx + 1}/{total} images @ longest_edge={longest_edge}")
    return cache


def cam_to_viewmat(cam):
    W2C = np.eye(4)
    W2C[:3, :3] = cam.R.T
    W2C[:3, 3] = cam.T
    return torch.tensor(W2C, dtype=torch.float32, device="cuda")


def cam_to_K(cam, width, height):
    tanfovx = math.tan(cam.FovX * 0.5)
    tanfovy = math.tan(cam.FovY * 0.5)
    fx = width / (2 * tanfovx)
    fy = height / (2 * tanfovy)
    return torch.tensor([
        [fx, 0, width / 2.0],
        [0, fy, height / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device="cuda")


# ═════════════════════════════════════════════════════════════════
# LR warmup helper
# ═════════════════════════════════════════════════════════════════

def warmup_lr_scale(iteration, warmup_iters):
    """Linear warmup from 0.1 to 1.0 over warmup_iters."""
    if warmup_iters <= 0 or iteration >= warmup_iters:
        return 1.0
    return 0.1 + 0.9 * (iteration / warmup_iters)


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )
    return is_distributed, rank, local_rank, world_size


def _cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def _all_reduce_optimizer_grads(optimizers, world_size):
    if world_size <= 1:
        return
    seen = set()
    for optimizer in optimizers:
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if param.grad is None or id(param) in seen:
                    continue
                seen.add(id(param))
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)


def _all_grads_finite(params) -> bool:
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is not None and not torch.isfinite(grad).all():
            return False
    return True


def _all_params_finite(params) -> bool:
    for param in params:
        if not torch.isfinite(param.data).all():
            return False
    return True


def _teacher_projection_state(teacher):
    if teacher is None:
        return None
    if hasattr(teacher, "get_projection_state"):
        return teacher.get_projection_state()
    return {
        "fine": teacher.proj_fine.state_dict(),
        "coarse": teacher.proj_coarse.state_dict(),
    }


def _load_teacher_projection_state(teacher, state):
    if teacher is None or state is None:
        return
    if hasattr(teacher, "load_projection_state"):
        teacher.load_projection_state(state)
    else:
        teacher.proj_fine.load_state_dict(state["fine"])
        teacher.proj_coarse.load_state_dict(state["coarse"])


def _infer_explicit_feature_root(init_ply: str | None) -> Path | None:
    if not init_ply:
        return None
    ply_path = Path(init_ply).expanduser()
    if not ply_path.is_absolute():
        ply_path = (Path.cwd() / ply_path).resolve()
    candidates = []
    if ply_path.parent.name.startswith("iteration_") or ply_path.parent.name == "best":
        candidates.append(ply_path.parent.parent.parent)
    candidates.append(ply_path.parent.parent)
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "features_best").exists() or (candidate / "features").exists():
            return candidate
    return None


def _resolve_explicit_feature_checkpoint(feature_root: Path, scale_name: str) -> Path | None:
    candidates = [
        feature_root / "features_best" / scale_name / "best_model.pth",
        feature_root / "features" / scale_name / "best_model.pth",
        feature_root / "features_best" / scale_name / "latest_model.pth",
        feature_root / "features" / scale_name / "latest_model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_explicit_loc_feature(path: Path) -> torch.Tensor:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or "loc_feature" not in payload:
        raise KeyError(f"{path} does not contain loc_feature")
    feat = payload["loc_feature"]
    if not torch.is_tensor(feat) or feat.ndim != 2:
        raise ValueError(f"Expected 2D loc_feature tensor in {path}")
    return feat.float()


def _maybe_init_latent_from_explicit_features(
    gaussians: HybridGaussianModel,
    init_ply: str | None,
    tcfg: dict,
) -> bool:
    scale_names = tcfg.get("init_latent_from_explicit_scales")
    if not scale_names:
        return False
    if isinstance(scale_names, str):
        scale_names = [scale_names]

    feature_root_cfg = tcfg.get("explicit_feature_root")
    feature_root = (
        Path(feature_root_cfg).expanduser()
        if feature_root_cfg
        else _infer_explicit_feature_root(init_ply)
    )
    if feature_root is None:
        print("  [LatentInit] Could not infer explicit feature root; skipping latent warm start")
        return False
    if not feature_root.is_absolute():
        feature_root = (Path.cwd() / feature_root).resolve()
    if not feature_root.exists():
        print(f"  [LatentInit] Explicit feature root not found: {feature_root}")
        return False

    explicit_features = []
    loaded_scales = []
    for scale_name in scale_names:
        ckpt_path = _resolve_explicit_feature_checkpoint(feature_root, scale_name)
        if ckpt_path is None:
            print(f"  [LatentInit] Missing explicit feature checkpoint for {scale_name} under {feature_root}")
            continue
        feature = _load_explicit_loc_feature(ckpt_path)
        if feature.shape[0] != gaussians.num_points:
            print(
                f"  [LatentInit] Shape mismatch for {scale_name}: "
                f"{tuple(feature.shape)} vs gaussians={gaussians.num_points}"
            )
            continue
        explicit_features.append(feature)
        loaded_scales.append(scale_name)

    if not explicit_features:
        print("  [LatentInit] No compatible explicit features found; skipping latent warm start")
        return False

    fused = torch.cat(explicit_features, dim=1)
    fused = fused - fused.mean(dim=0, keepdim=True)
    latent_dim = gaussians.latent_dim
    q = min(latent_dim, fused.shape[0], fused.shape[1])
    if q <= 0:
        print("  [LatentInit] Degenerate explicit feature tensor; skipping latent warm start")
        return False

    try:
        _, singular_values, basis = torch.pca_lowrank(fused, q=q, center=False)
        projected = fused @ basis[:, :q]
    except RuntimeError:
        _, singular_values, vh = torch.linalg.svd(fused, full_matrices=False)
        basis = vh[:q].transpose(0, 1)
        projected = fused @ basis

    projected = projected[:, :q]
    proj_mean = projected.mean(dim=0, keepdim=True)
    proj_std = projected.std(dim=0, keepdim=True).clamp(min=1e-6)
    projected = (projected - proj_mean) / proj_std

    if q < latent_dim:
        projected = torch.cat(
            [projected, torch.zeros(projected.shape[0], latent_dim - q, dtype=projected.dtype)],
            dim=1,
        )

    latent_scale = float(tcfg.get("init_latent_scale", 0.1))
    projected = projected[:, :latent_dim] * latent_scale
    gaussians._latent.data.copy_(projected.to(device=gaussians._latent.device, dtype=gaussians._latent.dtype))

    var_explained = float((singular_values[:q] ** 2).sum() / (singular_values ** 2).sum().clamp(min=1e-6))
    print(
        f"  [LatentInit] Initialized {latent_dim}d latent from explicit scales {loaded_scales} "
        f"(source_dim={fused.shape[1]}, pca_var={var_explained:.3f}, scale={latent_scale:.3f})"
    )
    return True


# ═════════════════════════════════════════════════════════════════
# Main Training
# ═════════════════════════════════════════════════════════════════

def train(cfg, resume_path=None):
    is_distributed, rank, local_rank, world_size = _init_distributed()
    is_main_process = rank == 0

    exp_name = cfg['exp_name']
    output_dir = os.path.join(cfg['output_dir'], exp_name)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    dcfg = cfg['dataset']
    mcfg = cfg['model']
    tcfg = cfg['training']
    hcfg = cfg['hash_grid']
    fcfg = cfg.get('fine_decoder', {})
    ccfg = cfg.get('coarse_decoder', {})
    teacher_cfg = cfg.get('teacher', {})
    freeze_teacher_projection = bool(teacher_cfg.get('freeze_projection', False))

    teacher_mode = teacher_cfg.get('mode', 'cached')
    seed = int(tcfg.get('seed', 12345)) + rank * 100003
    np.random.seed(seed)
    torch.manual_seed(seed)

    fine_feature_dim = int(mcfg.get('fine_feature_dim', mcfg.get('feature_dim', 64)))
    coarse_feature_dim = int(mcfg.get('coarse_feature_dim', mcfg.get('feature_dim', 64)))
    fine_latent_dim = mcfg.get('fine_latent_dim')
    coarse_latent_dim = mcfg.get('coarse_latent_dim')
    if fine_latent_dim is not None:
        fine_latent_dim = int(fine_latent_dim)
    if coarse_latent_dim is not None:
        coarse_latent_dim = int(coarse_latent_dim)

    if is_main_process:
        print(f"\n{'='*70}")
        print(f"  DCFF v5: Feature Field Training")
        print(f"  Experiment: {exp_name}")
        print(f"{'='*70}")
        print(f"  Teacher:        {teacher_mode}")
        print(f"  Distributed:    {world_size} GPU(s), local_rank={local_rank}")
        print(f"  Dataset:        {dcfg['source_dir']}")
        print(f"  Latent dim:     {mcfg['latent_dim']} "
              f"(fine={fine_latent_dim or mcfg['latent_dim']}, "
              f"coarse={coarse_latent_dim or mcfg['latent_dim']})")
        print(f"  Feature dims:   fine={fine_feature_dim}, coarse={coarse_feature_dim}")
        print(f"  Iterations:     {tcfg['iterations']}")
        print(f"  Fine start:     {tcfg['fine_start_iter']}")
        print(f"  Coarse start:   {tcfg['coarse_start_iter']}")
        print(f"{'='*70}\n")

        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

    # ── 1. Load scene ──
    print("Loading scene...")
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
        load_scene_colmap(dcfg['source_dir'], dcfg.get('images', ''))

    # ── 2. Setup teacher ──
    print("\nSetting up teacher...")

    if teacher_mode in {'online', 'online_bottleneck'}:
        teacher_input_longest_edge = int(
            teacher_cfg.get('input_longest_edge', tcfg.get('longest_edge', 960))
        )
        pca_init_dir = teacher_cfg.get('pca_init_dir', None)
        if pca_init_dir is None:
            candidate = os.path.join(dcfg.get('feature_dir', ''), 'pca_params')
            pca_init_dir = candidate if os.path.isdir(candidate) else None
        teacher = OnlineRadioTeacher(
            target_dim=mcfg.get('feature_dim', fine_feature_dim),
            fine_dim=fine_feature_dim,
            coarse_dim=coarse_feature_dim,
            bottleneck=(teacher_mode == 'online_bottleneck'),
            shallow_block=teacher_cfg.get('shallow_block', 10),
            radio_repo=teacher_cfg.get('radio_repo', 'feature_extract/checkpoints/RADIO'),
            pca_init_dir=pca_init_dir,
            compress_hidden_dim=int(teacher_cfg.get('compress', {}).get('hidden_dim', 256)),
            sample_pixels=int(teacher_cfg.get('compress', {}).get('sample_pixels', 1024)),
            recon_chunk_pixels=int(teacher_cfg.get('compress', {}).get('recon_chunk_pixels', 4096)),
            recon_cos_weight=float(teacher_cfg.get('compress', {}).get('recon_cos_weight', 1.0)),
            recon_l1_weight=float(teacher_cfg.get('compress', {}).get('recon_l1_weight', 0.25)),
            min_spatial_std=float(teacher_cfg.get('compress', {}).get('min_spatial_std', 0.0)),
            std_weight=float(teacher_cfg.get('compress', {}).get('std_weight', 0.0)),
            decorrelation_weight=float(teacher_cfg.get('compress', {}).get('decorrelation_weight', 0.0)),
            fine_raw_highpass_kernel=int(teacher_cfg.get('compress', {}).get('fine_raw_highpass_kernel', 0)),
            coarse_raw_highpass_kernel=int(teacher_cfg.get('compress', {}).get('coarse_raw_highpass_kernel', 0)),
            fine_adapter_highpass_kernel=int(teacher_cfg.get('compress', {}).get('fine_adapter_highpass_kernel', 0)),
            coarse_adapter_highpass_kernel=int(teacher_cfg.get('compress', {}).get('coarse_adapter_highpass_kernel', 0)),
            normalize_output=bool(teacher_cfg.get('compress', {}).get('normalize_output', True)),
        ).cuda()
        # Compute feature resolution from actual image dimensions
        sample_cam = train_cams[0]
        img_w, img_h = sample_cam.width, sample_cam.height
        scale = teacher_input_longest_edge / max(img_w, img_h)
        if scale < 1.0:
            img_w, img_h = int(img_w * scale), int(img_h * scale)
        teacher.set_image_size(img_h, img_w)
        feat_h, feat_w = teacher.feature_resolution
        print(f"  Image size: {img_w}×{img_h} → RADIO features: {feat_w}×{feat_h}")
        radio_cache = None
        cam_to_fid = None
    else:
        teacher = None
        radio_cache = CachedFeatureTeacher(dcfg['feature_dir'])
        feat_h, feat_w = radio_cache.feat_h, radio_cache.feat_w
        if (fine_feature_dim, coarse_feature_dim) != (radio_cache.feature_dim, radio_cache.feature_dim):
            raise ValueError(
                "cached teacher only supports equal cached dimensions. "
                f"Config requested fine={fine_feature_dim}, coarse={coarse_feature_dim}, "
                f"cache has {radio_cache.feature_dim}d. Use teacher.mode=online_bottleneck."
            )

        # Build camera → frame ID mapping
        images_subdir = dcfg.get('images', '')
        if images_subdir:
            images_dir = os.path.join(dcfg['source_dir'], images_subdir)
        else:
            import glob as _glob
            if _glob.glob(os.path.join(dcfg['source_dir'], 'seq*')):
                images_dir = dcfg['source_dir']
            elif os.path.isdir(os.path.join(dcfg['source_dir'], 'images')):
                images_dir = os.path.join(dcfg['source_dir'], 'images')
            else:
                images_dir = dcfg['source_dir']
        da3_name_to_fid = build_da3_image_order(images_dir)

        cam_to_fid = {}
        for cam in train_cams:
            fid = da3_name_to_fid.get(cam.image_name)
            if fid is not None and fid in radio_cache.frame_ids:
                cam_to_fid[cam.uid] = fid

        # Also map test cameras for validation
        for cam in test_cams:
            fid = da3_name_to_fid.get(cam.image_name)
            if fid is not None and fid in radio_cache.frame_ids:
                cam_to_fid[cam.uid] = fid
        n_test_mapped = sum(1 for c in test_cams if c.uid in cam_to_fid)
        print(f"  Matched {len(cam_to_fid) - n_test_mapped}/{len(train_cams)} train + "
              f"{n_test_mapped}/{len(test_cams)} test cameras to features")

    # Coarse feature resolution (half of fine)
    coarse_h = feat_h // 2
    coarse_w = feat_w // 2
    print(f"  Fine resolution:   {feat_w}×{feat_h}")
    print(f"  Coarse resolution: {coarse_w}×{coarse_h}")

    # ── 3. Create models ──
    print("\nInitializing models...")
    init_ply = tcfg.get('init_ply', None)
    warmstart_path = tcfg.get('warmstart_checkpoint')

    gaussians = HybridGaussianModel(
        sh_degree=mcfg['sh_degree'],
        latent_dim=mcfg['latent_dim'],
    )

    train_args = argparse.Namespace(**{
        'position_lr_init': float(tcfg['position_lr_init']),
        'position_lr_final': float(tcfg['position_lr_final']),
        'feature_lr': float(tcfg['feature_lr']),
        'opacity_lr': float(tcfg['opacity_lr']),
        'scaling_lr': float(tcfg['scaling_lr']),
        'rotation_lr': float(tcfg['rotation_lr']),
        'latent_lr': float(tcfg['latent_lr']),
        'percent_dense': float(tcfg['percent_dense']),
        'iterations': tcfg['iterations'],
    })

    # Load pretrained geometry
    freeze_geometry = tcfg.get('freeze_geometry', True)
    if init_ply:
        gaussians.load_ply(init_ply, freeze_geometry=freeze_geometry)
        gaussians.spatial_lr_scale = cameras_extent
        print(f"  Loaded PLY: {init_ply}")
        print(f"  Gaussians: {gaussians.num_points:,}, geometry frozen={freeze_geometry}")
    else:
        gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)
    _maybe_init_latent_from_explicit_features(gaussians, init_ply, tcfg)
    gaussians.training_setup(train_args)

    bg_color = torch.tensor(
        [1, 1, 1] if mcfg.get('white_background', False) else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )

    # Scene extent for hash grid
    xyz_init = gaussians.get_xyz.detach().cpu().numpy()
    scene_extent = float(np.percentile(np.linalg.norm(xyz_init, axis=1), 99)) * 1.2
    print(f"  Scene extent: {scene_extent:.2f}")

    hash_latent_dim = coarse_latent_dim if coarse_latent_dim is not None else mcfg['latent_dim']

    # Hash Grid
    hash_grid = SpatialHashGrid(
        scene_extent=scene_extent,
        feature_dim=coarse_feature_dim,
        input_mode=hcfg.get('input_mode', 'implicit_scale'),
        latent_dim=hash_latent_dim,
        scale_dim=hcfg.get('scale_dim', 2),
        scale_pe_freqs=hcfg.get('scale_pe_freqs', 4),
        include_raw_scale=hcfg.get('include_raw_scale', True),
        n_levels=hcfg['n_levels'],
        n_features_per_level=hcfg['n_features_per_level'],
        log2_hashmap_size=hcfg['log2_hashmap_size'],
        base_resolution=hcfg['base_resolution'],
        max_resolution=hcfg['max_resolution'],
        sh_degree=hcfg.get('sh_degree', 3),
        mlp_hidden=hcfg.get('mlp_hidden', 256),
        mlp_layers=hcfg.get('mlp_layers', 4),
        forward_chunk_size=hcfg.get('forward_chunk_size', 0),
    ).cuda()

    # Deferred Cascaded Renderer
    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid,
        latent_dim=mcfg['latent_dim'],
        fine_latent_dim=fine_latent_dim,
        coarse_latent_dim=coarse_latent_dim,
        fine_feature_dim=fine_feature_dim,
        coarse_feature_dim=coarse_feature_dim,
        fine_hidden_dim=fcfg.get('hidden_dim', 256),
        fine_num_layers=fcfg.get('num_layers', 5),
        fine_use_viewdirs=fcfg.get('use_viewdirs', True),
        fine_view_degree=fcfg.get('view_degree', 2),
        fine_decoder_type=fcfg.get('type', 'pointwise'),
        coarse_mode=ccfg.get('mode', 'implicit_only'),
        coarse_carrier_hidden_dim=ccfg.get('carrier_hidden_dim'),
        coarse_gate_hidden_dim=ccfg.get('gate_hidden_dim'),
        coarse_forward_batch_chunk_size=ccfg.get('forward_batch_chunk_size', 0),
        coarse_smoothing_kernel=mcfg.get('coarse_smoothing_kernel', 1),
    ).cuda()

    # ── 3b. Optional screen-space refiner ──
    refiner_cfg = cfg.get('refiner', {})
    refiner_type = refiner_cfg.get('type')
    feat_sharp_fine = None
    if refiner_type == 'depth_guided':
        refiner_hdim = int(refiner_cfg.get('hidden_dim', 128))
        feat_sharp_fine = DepthGuidedRefiner(
            fine_feature_dim, hidden_dim=refiner_hdim,
        ).cuda()

    # ── 3c. Feature Selection Module ──
    fsm_cfg = cfg.get('fsm', {})
    use_fsm = fsm_cfg.get('enable', False)
    feat_select = None
    if use_fsm:
        if fine_feature_dim != coarse_feature_dim:
            raise ValueError("FSM requires equal fine/coarse feature dims; disable fsm for asymmetric DCFF")
        from feature_field.dcff.feature_selection import FeatureSelectionModule
        feat_select = FeatureSelectionModule(
            feature_dim=fine_feature_dim,
            hidden_dim=int(fsm_cfg.get('hidden_dim', 32)),
            num_heads=int(fsm_cfg.get('num_heads', 4)),
            channel_routing_mode=fsm_cfg.get('channel_routing_mode', 'categorical'),
            use_channel_select=fsm_cfg.get('use_channel_select', True),
            use_cross_attn=fsm_cfg.get('use_cross_attn', True),
            use_spatial_conf=fsm_cfg.get('use_spatial_conf', True),
        ).cuda()
        n_fsm_local = sum(p.numel() for p in feat_select.parameters())
        print(f"  FSM: {n_fsm_local:,} params")

    if teacher is not None and freeze_teacher_projection:
        for param in teacher.get_projection_params():
            param.requires_grad_(False)
        print("  Teacher projection/compressor frozen")

    # Loss
    loss_cfg = tcfg.get('loss', {})
    loss_fn = DCFFLoss(
        lambda_dssim=float(loss_cfg.get('lambda_dssim', 0.2)),
        lambda_fine_cos=float(loss_cfg.get('lambda_fine_cos', 1.0)),
        lambda_fine_l1=float(loss_cfg.get('lambda_fine_l1', 1.0)),
        lambda_coarse_cos=float(loss_cfg.get('lambda_coarse_cos', 1.0)),
        lambda_coarse_l1=float(loss_cfg.get('lambda_coarse_l1', 1.0)),
        lambda_tv=float(loss_cfg.get('lambda_tv', 0.02)),
        lambda_coarse_screen_tv=float(loss_cfg.get('lambda_coarse_screen_tv', 0.0)),
        lambda_normal=float(loss_cfg.get('lambda_normal', 0.0)),
        lambda_dist=float(loss_cfg.get('lambda_dist', 0.0)),
        lambda_channel_std=float(loss_cfg.get('lambda_channel_std', 0.1)),
        lambda_fine_grad=float(loss_cfg.get('lambda_fine_grad', 0.0)),
        lambda_fine_edge=float(loss_cfg.get('lambda_fine_edge', 0.0)),
        fine_edge_strength=float(loss_cfg.get('fine_edge_strength', 2.0)),
        fine_edge_dilation=int(loss_cfg.get('fine_edge_dilation', 3)),
        lambda_coarse_grad=float(loss_cfg.get('lambda_coarse_grad', 0.0)),
        lambda_fine_nce=float(loss_cfg.get('lambda_fine_nce', 0.0)),
        lambda_coarse_nce=float(loss_cfg.get('lambda_coarse_nce', 0.0)),
        nce_temperature=float(loss_cfg.get('nce_temperature', 0.07)),
        nce_samples=int(loss_cfg.get('nce_samples', 512)),
    )

    # Print param counts
    n_hash = sum(p.numel() for p in hash_grid.parameters())
    n_fine = sum(p.numel() for p in renderer.fine_decoder.parameters())
    n_coarse_fusion = sum(p.numel() for p in renderer.coarse_carrier_fusion.parameters()) if renderer.coarse_carrier_fusion is not None else 0
    n_refiner = sum(p.numel() for p in feat_sharp_fine.parameters()) if feat_sharp_fine is not None else 0
    n_fsm = sum(p.numel() for p in feat_select.parameters()) if feat_select is not None else 0
    n_proj = 0
    if teacher is not None:
        n_proj = sum(p.numel() for p in teacher.get_projection_params() if p.requires_grad)
    print(f"\n  Gaussians:    {gaussians.num_points:,}")
    print(f"  Latent dim:   {mcfg['latent_dim']}")
    print(f"  Hash grid:    {n_hash:,} params")
    print(f"  Fine decoder: {n_fine:,} params")
    if n_coarse_fusion > 0:
        print(f"  Coarse fuse:  {n_coarse_fusion:,} params ({ccfg.get('mode', 'implicit_only')})")
    if n_refiner > 0:
        print(f"  Refiner:      {n_refiner:,} params (DepthGuided, hidden={refiner_cfg.get('hidden_dim', 128)})")
    if n_proj > 0:
        print(f"  Projections:  {n_proj:,} params")
    if n_fsm > 0:
        print(f"  FSM:          {n_fsm:,} params")
    print(f"  Total DCFF:   {n_hash + n_fine + n_coarse_fusion + n_refiner + n_fsm + n_proj:,} trainable\n")

    # ── 4. Optimizers ──
    dcff_params = [
        {'params': hash_grid.parameters(), 'lr': float(tcfg['lr_hash_grid'])},
        {'params': renderer.fine_decoder.parameters(), 'lr': float(tcfg['lr_fine_decoder'])},
    ]
    if renderer.coarse_carrier_fusion is not None:
        dcff_params.append({
            'params': renderer.coarse_carrier_fusion.parameters(),
            'lr': float(tcfg.get('lr_coarse_fusion', tcfg.get('lr_fine_decoder', 0.0003))),
        })
    if feat_sharp_fine is not None:
        dcff_params.append({
            'params': feat_sharp_fine.parameters(),
            'lr': float(tcfg.get('lr_refiner', tcfg.get('lr_fine_decoder', 0.0003))),
        })
    if feat_select is not None:
        dcff_params.append({
            'params': feat_select.parameters(),
            'lr': float(fsm_cfg.get('lr', 0.0003)),
        })
    if teacher is not None and not freeze_teacher_projection and float(teacher_cfg.get('lr_projection', 0.0005)) > 0:
        dcff_params.append({
            'params': teacher.get_projection_params(),
            'lr': float(teacher_cfg.get('lr_projection', 0.0005)),
        })

    dcff_optimizer = torch.optim.Adam(dcff_params, eps=1e-15)
    dcff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dcff_optimizer,
        T_max=tcfg['iterations'] - tcfg['fine_start_iter'],
        eta_min=1e-6,
    )

    # Mixed precision training for speed (bfloat16 for stability with large models)
    use_amp = tcfg.get('use_amp', True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 4b. Optional warm-start (weights only, no optimizer/iteration restore) ──
    if warmstart_path and os.path.exists(warmstart_path):
        restore_warmstart_geometry = bool(tcfg.get('warmstart_restore_geometry_state', True))
        restore_warmstart_coarse_fusion = bool(tcfg.get('warmstart_restore_coarse_fusion_state', True))
        restore_warmstart_fine_decoder = bool(tcfg.get('warmstart_restore_fine_decoder_state', True))
        warm_ckpt = safe_torch_load(warmstart_path, map_location='cuda')
        hash_grid.load_state_dict(warm_ckpt['hash_grid_state'])
        if restore_warmstart_fine_decoder and 'fine_decoder_state' in warm_ckpt:
            try:
                renderer.fine_decoder.load_state_dict(warm_ckpt['fine_decoder_state'])
            except RuntimeError as exc:
                print(f"  Skipped warm-start fine decoder state due to mismatch: {exc}")
        elif 'fine_decoder_state' in warm_ckpt:
            print("  Skipped warm-start fine decoder state")
        if (
            renderer.coarse_carrier_fusion is not None
            and 'coarse_fusion_state' in warm_ckpt
            and restore_warmstart_coarse_fusion
        ):
            renderer.coarse_carrier_fusion.load_state_dict(warm_ckpt['coarse_fusion_state'])
        elif renderer.coarse_carrier_fusion is not None and 'coarse_fusion_state' in warm_ckpt:
            print("  Skipped warm-start coarse fusion state")
        if feat_sharp_fine is not None and 'feat_sharp_fine_state' in warm_ckpt:
            feat_sharp_fine.load_state_dict(warm_ckpt['feat_sharp_fine_state'])
        if feat_select is not None and 'fsm_state' in warm_ckpt:
            try:
                feat_select.load_state_dict(warm_ckpt['fsm_state'], strict=False)
            except RuntimeError as exc:
                print(f"  Skipped FSM warm-start due to mismatch: {exc}")
        restore_warmstart_projection = bool(tcfg.get('warmstart_restore_projection_state', True))
        if 'projection_state' in warm_ckpt and teacher is not None and restore_warmstart_projection:
            _load_teacher_projection_state(teacher, warm_ckpt['projection_state'])
        elif 'projection_state' in warm_ckpt and teacher is not None:
            print("  Skipped warm-start teacher projection state")
        restored_latent = False
        if restore_warmstart_geometry and 'gaussians_state' in warm_ckpt:
            gs = warm_ckpt['gaussians_state']
            gaussians._xyz.data.copy_(gs['_xyz'].cuda())
            gaussians._rotation.data.copy_(gs['_rotation'].cuda())
            gaussians._scaling.data.copy_(gs['_scaling'].cuda())
            gaussians._opacity.data.copy_(gs['_opacity'].cuda())
            gaussians._features_dc.data.copy_(gs['_features_dc'].cuda())
            gaussians._features_rest.data.copy_(gs['_features_rest'].cuda())
            gaussians.active_sh_degree = gs['active_sh_degree']
            if 'latent' in warm_ckpt and warm_ckpt['latent'].shape == gaussians._latent.shape:
                gaussians._latent.data.copy_(warm_ckpt['latent'].cuda())
                restored_latent = True
            print("  Restored warm-start Gaussian state")
        elif 'latent' in warm_ckpt and warm_ckpt['latent'].shape == gaussians._latent.shape:
            gaussians._latent.data.copy_(warm_ckpt['latent'].cuda())
            restored_latent = True
        if not restore_warmstart_geometry and 'gaussians_state' in warm_ckpt:
            print("  Skipped warm-start Gaussian state to preserve init_ply geometry")
        if restored_latent:
            print("  Restored warm-start latent")
        print(f"  Warm-started weights from {warmstart_path}")

    # ── 5. Resume ──
    start_iteration = 0
    best_metrics = {'total_loss': float('inf')}

    if resume_path and os.path.exists(resume_path):
        ckpt = safe_torch_load(resume_path, map_location='cuda')
        hash_grid.load_state_dict(ckpt['hash_grid_state'])
        renderer.fine_decoder.load_state_dict(ckpt['fine_decoder_state'])
        if renderer.coarse_carrier_fusion is not None and 'coarse_fusion_state' in ckpt:
            renderer.coarse_carrier_fusion.load_state_dict(ckpt['coarse_fusion_state'])
        if feat_sharp_fine is not None and 'feat_sharp_fine_state' in ckpt:
            feat_sharp_fine.load_state_dict(ckpt['feat_sharp_fine_state'])
        if feat_select is not None and 'fsm_state' in ckpt:
            feat_select.load_state_dict(ckpt['fsm_state'])
        if 'projection_state' in ckpt and teacher is not None:
            _load_teacher_projection_state(teacher, ckpt['projection_state'])
        dcff_optimizer.load_state_dict(ckpt['dcff_optimizer_state'])
        dcff_scheduler.load_state_dict(ckpt['dcff_scheduler_state'])
        if 'gaussians_optimizer_state' in ckpt:
            gaussians.optimizer.load_state_dict(ckpt['gaussians_optimizer_state'])
        if 'gaussians_state' in ckpt:
            gs = ckpt['gaussians_state']
            gaussians._xyz.data.copy_(gs['_xyz'].cuda())
            gaussians._rotation.data.copy_(gs['_rotation'].cuda())
            gaussians._scaling.data.copy_(gs['_scaling'].cuda())
            gaussians._opacity.data.copy_(gs['_opacity'].cuda())
            gaussians._features_dc.data.copy_(gs['_features_dc'].cuda())
            gaussians._features_rest.data.copy_(gs['_features_rest'].cuda())
            gaussians.active_sh_degree = gs['active_sh_degree']
            if 'latent' in ckpt and ckpt['latent'].shape == gaussians._latent.shape:
                gaussians._latent.data.copy_(ckpt['latent'].cuda())
            print(f"  Restored full gaussians state (xyz, rot, scale, opacity, sh)")
        start_iteration = ckpt['iteration']
        best_metrics = ckpt.get('best_metrics', best_metrics)
        print(f"  Resumed from iteration {start_iteration}")

    # ── 6. Training loop ──
    longest_edge = tcfg.get('longest_edge', 960)
    batch_size = int(tcfg.get('batch_size', 4))
    total_iters = tcfg['iterations']
    fine_start = tcfg['fine_start_iter']
    coarse_start = tcfg['coarse_start_iter']
    coarse_save_start = int(tcfg.get('coarse_start_iter', tcfg.get('coarse_start', 2000)))
    warmup_iters = int(tcfg.get('warmup_iters', 1000))
    grad_clip = float(tcfg.get('grad_clip', 1.0))
    log_every = tcfg.get('log_every', 100)
    save_every = tcfg.get('save_every', 5000)
    val_every = tcfg.get('val_every', 5000)
    sh_up_every = tcfg.get('sh_degree_up_every', 1000)
    coarse_downsample = tcfg.get('coarse_downsample', True)
    best_save_every = max(1, int(tcfg.get('best_save_every', 1)))
    max_bad_steps = max(1, int(tcfg.get('max_bad_steps', 20)))
    teacher_input_longest_edge = int(
        teacher_cfg.get('input_longest_edge', tcfg.get('longest_edge', 960))
    )
    teacher_micro_batch = int(teacher_cfg.get('micro_batch', tcfg.get('teacher_micro_batch', batch_size)))
    compress_loss_weight = float(teacher_cfg.get('compress', {}).get('loss_weight', 1.0))

    # Valid cameras
    if teacher_mode == 'cached':
        valid_cams = [c for c in train_cams if c.uid in cam_to_fid]
    else:
        valid_cams = list(train_cams)
    n_cams = len(valid_cams)

    image_cache = None
    teacher_image_cache = None
    if tcfg.get('cache_resized_images', False):
        cache_dtype_name = str(tcfg.get('image_cache_dtype', 'float16')).lower()
        cache_dtype = torch.float32 if cache_dtype_name == 'float32' else torch.float16
        log_msg = f"Pre-caching GT images on GPU ({cache_dtype_name}, longest_edge={longest_edge})"
        if is_main_process:
            print(log_msg)
        image_cache = _cache_resized_images(
            valid_cams, longest_edge, dtype=cache_dtype, show_progress=is_main_process,
        )
    if teacher_mode in {'online', 'online_bottleneck'} and teacher_cfg.get(
        'cache_resized_images', tcfg.get('cache_resized_images', False)
    ):
        if teacher_input_longest_edge == longest_edge and image_cache is not None:
            teacher_image_cache = image_cache
        else:
            cache_dtype_name = str(teacher_cfg.get('image_cache_dtype', tcfg.get('image_cache_dtype', 'float16'))).lower()
            cache_dtype = torch.float32 if cache_dtype_name == 'float32' else torch.float16
            if is_main_process:
                print(
                    f"Pre-caching teacher images on GPU ({cache_dtype_name}, "
                    f"longest_edge={teacher_input_longest_edge})"
                )
            teacher_image_cache = _cache_resized_images(
                valid_cams, teacher_input_longest_edge, dtype=cache_dtype,
                show_progress=is_main_process,
            )

    log_path = os.path.join(output_dir, 'train.log')
    log_f = open(log_path, 'a' if start_iteration > 0 else 'w') if is_main_process else None

    def log(msg):
        if not is_main_process:
            return
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f"Training {total_iters} iters, {n_cams} cameras, batch_per_gpu={batch_size}, "
        f"global_batch={batch_size * world_size}")
    log(f"Teacher: {teacher_mode}, fine_dim={fine_feature_dim}, coarse_dim={coarse_feature_dim}, "
        f"latent_dim={mcfg['latent_dim']}")
    log(f"Fine: {feat_w}×{feat_h}, Coarse: {coarse_w}×{coarse_h}")
    log(f"Warmup: {warmup_iters} iters, grad_clip: {grad_clip}")

    start_time = time.time()
    consecutive_bad_steps = 0

    for iteration in range(start_iteration + 1, total_iters + 1):
        # Phase
        if iteration < fine_start:
            phase = 1
        elif iteration < coarse_start:
            phase = 2
        else:
            phase = 3

        gaussians.update_learning_rate(iteration)

        if iteration % sh_up_every == 0:
            gaussians.oneupSHdegree()

        # Store initial LR on first iteration
        if iteration == start_iteration + 1:
            for pg in dcff_optimizer.param_groups:
                pg['initial_lr'] = pg['lr']

        # LR warmup (applied before scheduler)
        if warmup_iters > 0 and iteration <= warmup_iters + start_iteration:
            scale = warmup_lr_scale(iteration - start_iteration, warmup_iters)
            for pg in dcff_optimizer.param_groups:
                if 'initial_lr' in pg:
                    pg['lr'] = pg['initial_lr'] * scale

        # Sample batch
        batch_indices = np.random.randint(n_cams, size=batch_size)
        cams = [valid_cams[idx] for idx in batch_indices]

        # Load GT image for rendering
        gt_image = load_image_batch(cams, longest_edge, cache=image_cache)
        _, _, img_h, img_w = gt_image.shape

        # Camera matrices
        viewmat = torch.stack([cam_to_viewmat(c) for c in cams], dim=0)
        K = torch.stack([cam_to_K(c, img_w, img_h) for c in cams], dim=0)

        # ── Get teacher targets before rendering ──
        # RADIO forward has a large peak even under no_grad.  Preparing targets
        # before the 2DGS render graph exists prevents the two peaks from
        # stacking and causing late-iteration OOMs.
        fine_target = None
        coarse_target = None
        compression_losses = {}

        if phase >= 2:
            if teacher_mode in {'online', 'online_bottleneck'}:
                teacher_images = load_image_batch(
                    cams, teacher_input_longest_edge, cache=teacher_image_cache,
                )
                fine_proj, coarse_proj, compression_losses = _extract_project_teacher_batch(
                    teacher=teacher,
                    teacher_images=teacher_images,
                    teacher_mode=teacher_mode,
                    micro_batch=teacher_micro_batch,
                    return_compression_loss=(
                        teacher_mode == 'online_bottleneck'
                        and compress_loss_weight > 0
                        and not freeze_teacher_projection
                    ),
                )

                if teacher_cfg.get('detach_targets_for_field_loss', False):
                    fine_proj_for_loss = fine_proj.detach()
                    coarse_proj_for_loss = coarse_proj.detach()
                else:
                    fine_proj_for_loss = fine_proj
                    coarse_proj_for_loss = coarse_proj
                del teacher_images

                fine_target = fine_proj_for_loss  # [B, fine_dim, Hp, Wp]
                if phase >= 3:
                    if coarse_downsample:
                        coarse_target = F.interpolate(
                            coarse_proj_for_loss, (coarse_h, coarse_w),
                            mode='bilinear', align_corners=False)
                    else:
                        coarse_target = coarse_proj_for_loss
            else:
                fids = [cam_to_fid[c.uid] for c in cams]
                geo_batch, sem_batch = [], []
                for fid in fids:
                    geo, sem = radio_cache.get(fid)
                    geo_batch.append(geo)
                    sem_batch.append(sem)
                fine_target = torch.stack(geo_batch, dim=0)
                if phase >= 3:
                    coarse_sem = torch.stack(sem_batch, dim=0)
                    if coarse_downsample:
                        coarse_target = F.interpolate(
                            coarse_sem, (coarse_h, coarse_w),
                            mode='bilinear', align_corners=False)
                    else:
                        coarse_target = coarse_sem

        # ── Forward: Render ──
        with torch.cuda.amp.autocast(enabled=use_amp):
            render_result = renderer(
                gaussians,
                viewmat=viewmat,
                K=K,
                width=img_w,
                height=img_h,
                render_coarse=(phase >= 3),
                feature_height=feat_h,
                feature_width=feat_w,
            )

        # ── Apply screen-space refiner to fine features ──
        if feat_sharp_fine is not None and phase >= 2:
            depth_feat = F.interpolate(
                render_result['depth'], (feat_h, feat_w),
                mode='bilinear', align_corners=False)
            alpha_feat = F.interpolate(
                render_result['alpha'], (feat_h, feat_w),
                mode='bilinear', align_corners=False)
            render_result = dict(render_result)  # shallow copy to avoid mutating
            render_result['fine_features'] = feat_sharp_fine(
                render_result['fine_features'].float(),
                depth=depth_feat.float(), alpha=alpha_feat.float())

        # ── Feature Selection Module (FSM) ──
        if feat_select is not None and phase >= 2:
            depth_feat = F.interpolate(
                render_result['depth'], (feat_h, feat_w),
                mode='bilinear', align_corners=False)
            alpha_feat = F.interpolate(
                render_result['alpha'], (feat_h, feat_w),
                mode='bilinear', align_corners=False)
            render_result = dict(render_result)  # shallow copy to avoid mutating

            # Get coarse at fine resolution if needed
            if render_result.get('coarse_features') is not None:
                coarse_for_fsm = F.interpolate(
                    render_result['coarse_features'], (feat_h, feat_w),
                    mode='bilinear', align_corners=False)
            else:
                coarse_for_fsm = None

            # FSM temperature schedule
            fsm_temp = max(0.5, 2.0 - iteration / 10000)

            fsm_result = feat_select(
                fine_features=render_result['fine_features'],
                coarse_features=coarse_for_fsm if coarse_for_fsm is not None else render_result['fine_features'],
                alpha=alpha_feat,
                depth=depth_feat,
                temperature=fsm_temp,
            )
            render_result['fine_features'] = fsm_result['fine_features']
            if coarse_for_fsm is not None:
                render_result['coarse_features'] = fsm_result['coarse_features']
            render_result['fsm_spatial_conf'] = fsm_result['spatial_confidence']
            render_result['fsm_channel_weights'] = fsm_result['channel_weights']

        # ── Multi-resolution coarse: downsample prediction too ──
        coarse_pred_for_loss = None
        if phase >= 3 and coarse_downsample:
            coarse_features = render_result.get('coarse_features')
            if coarse_features is not None:
                coarse_pred_for_loss = F.interpolate(
                    coarse_features, (coarse_h, coarse_w),
                    mode='bilinear', align_corners=False)

        # ── Loss computation ──
        # For multi-res coarse: temporarily replace coarse features
        if coarse_pred_for_loss is not None:
            render_result_for_loss = dict(render_result)
            render_result_for_loss['coarse_features'] = coarse_pred_for_loss
        else:
            render_result_for_loss = render_result

        losses = loss_fn.compute(
            render_result=render_result_for_loss,
            gt_rgb=gt_image,
            radio_geo=fine_target,
            radio_sem=coarse_target,
            hash_grid=hash_grid if phase >= 3 else None,
            phase=phase,
        )

        total_loss = losses['total']
        if compression_losses:
            for name, value in compression_losses.items():
                losses[name] = value
            total_loss = total_loss + compress_loss_weight * compression_losses['compress_total']
            losses['total'] = total_loss

        # ── FSM auxiliary losses ──
        if feat_select is not None and phase >= 2:
            if render_result.get('fsm_spatial_conf') is not None:
                alpha_for_loss = F.interpolate(
                    render_result['alpha'], (feat_h, feat_w),
                    mode='bilinear', align_corners=False)
                fsm_conf = render_result['fsm_spatial_conf']
                fsm_conf_loss = F.mse_loss(fsm_conf, alpha_for_loss)
                losses['fsm_conf'] = fsm_conf_loss
                total_loss = total_loss + fsm_conf_loss * fsm_cfg.get('lambda_conf', 0.05)

        # ── NaN protection ──
        bad_step = torch.isnan(total_loss) or torch.isinf(total_loss)
        if is_distributed:
            bad_step_tensor = torch.tensor(
                [1 if bool(bad_step) else 0],
                device="cuda",
                dtype=torch.int32,
            )
            dist.all_reduce(bad_step_tensor, op=dist.ReduceOp.MAX)
            bad_step = bad_step_tensor.item() > 0

        if bad_step:
            consecutive_bad_steps += 1
            log(f"  [WARN] NaN/Inf at iter {iteration}, skipping step. "
                f"Components: " + ", ".join(
                    f"{k}={v.item():.4f}" if isinstance(v, torch.Tensor)
                    else f"{k}={v:.4f}" for k, v in losses.items()
                    if k != 'total'))
            gaussians.optimizer.zero_grad(set_to_none=True)
            if phase >= 2:
                dcff_optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.update()
            if consecutive_bad_steps >= max_bad_steps:
                raise RuntimeError(
                    f"Stopping after {consecutive_bad_steps} consecutive non-finite "
                    f"steps at iter {iteration}. Last safe checkpoint should be used."
                )
            continue

        # ── Backward + optimize ──
        scaler.scale(total_loss).backward()

        # Gradient clipping for ALL parameters
        all_grad_params = []
        for pg in gaussians.optimizer.param_groups:
            all_grad_params.extend(pg.get('params', []))
        if phase >= 2:
            all_grad_params += list(hash_grid.parameters()) + list(renderer.fine_decoder.parameters())
            if renderer.coarse_carrier_fusion is not None:
                all_grad_params += list(renderer.coarse_carrier_fusion.parameters())
            if feat_sharp_fine is not None:
                all_grad_params += list(feat_sharp_fine.parameters())
            if feat_select is not None:
                all_grad_params += list(feat_select.parameters())
            if teacher is not None:
                all_grad_params += teacher.get_projection_params()

        # Gradient clipping: unscale then clip, or clip directly
        if use_amp:
            scaler.unscale_(gaussians.optimizer)
            if phase >= 2:
                scaler.unscale_(dcff_optimizer)

        _all_reduce_optimizer_grads(
            [gaussians.optimizer, dcff_optimizer if phase >= 2 else None],
            world_size,
        )

        bad_grads = not _all_grads_finite(all_grad_params)
        if is_distributed:
            bad_grad_tensor = torch.tensor(
                [1 if bad_grads else 0],
                device="cuda",
                dtype=torch.int32,
            )
            dist.all_reduce(bad_grad_tensor, op=dist.ReduceOp.MAX)
            bad_grads = bad_grad_tensor.item() > 0

        if bad_grads:
            consecutive_bad_steps += 1
            log(f"  [WARN] Non-finite gradient at iter {iteration}, skipping optimizer step")
            gaussians.optimizer.zero_grad(set_to_none=True)
            if phase >= 2:
                dcff_optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.update()
            if consecutive_bad_steps >= max_bad_steps:
                raise RuntimeError(
                    f"Stopping after {consecutive_bad_steps} consecutive non-finite "
                    f"gradient steps at iter {iteration}. Last safe checkpoint should be used."
                )
            continue

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(all_grad_params, grad_clip)

        # Optimizer steps
        with torch.no_grad():
            # Always train geometry (joint training, not frozen)
            if use_amp:
                scaler.step(gaussians.optimizer)
            else:
                gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if phase >= 2:
                if use_amp:
                    scaler.step(dcff_optimizer)
                else:
                    dcff_optimizer.step()
                dcff_optimizer.zero_grad(set_to_none=True)
                dcff_scheduler.step()
            if use_amp:
                scaler.update()

        bad_params = not _all_params_finite(all_grad_params)
        if is_distributed:
            bad_param_tensor = torch.tensor(
                [1 if bad_params else 0],
                device="cuda",
                dtype=torch.int32,
            )
            dist.all_reduce(bad_param_tensor, op=dist.ReduceOp.MAX)
            bad_params = bad_param_tensor.item() > 0
        if bad_params:
            raise RuntimeError(
                f"Non-finite parameters after optimizer step at iter {iteration}; "
                "stop and resume from the last safe checkpoint with lower LR/loss weights."
            )

        consecutive_bad_steps = 0

        # ── Track best ──
        if iteration == coarse_save_start and best_metrics.get('iteration', 0) < coarse_save_start:
            best_metrics = {'total_loss': float('inf')}

        is_best = False
        can_update_best = (
            iteration >= coarse_save_start
            and (best_save_every <= 1 or iteration % best_save_every == 0 or iteration == total_iters)
        )
        if can_update_best and losses['total'].item() < best_metrics['total_loss']:
            is_best = True
            best_metrics['total_loss'] = losses['total'].item()
            best_metrics['iteration'] = iteration

        # ── Logging ──
        if iteration % log_every == 0:
            elapsed = time.time() - start_time
            elapsed_iters = max(1, iteration - start_iteration)
            it_s = elapsed_iters / max(elapsed, 1e-6)
            img_s = it_s * batch_size

            parts = [f"[Iter {iteration}]", f"P{phase}",
                     f"L={losses['total']:.4f}",
                     f"RGB={losses['rgb']:.4f}"]

            if 'fine_cos' in losses:
                parts.append(f"fine_cos={1 - losses['fine_cos'].item():.3f}")
            if 'fine_cs' in losses:
                parts.append(f"fine_cs={losses['fine_cs'].item():.4f}")
            if 'fine_grad' in losses and loss_cfg.get('lambda_fine_grad', 0.0) > 0:
                parts.append(f"fine_grad={losses['fine_grad'].item():.4f}")
            if 'fine_edge' in losses and loss_cfg.get('lambda_fine_edge', 0.0) > 0:
                parts.append(f"fine_edge={losses['fine_edge'].item():.4f}")
            if 'coarse_cos' in losses:
                parts.append(f"coarse_cos={1 - losses['coarse_cos'].item():.3f}")
            if 'coarse_cs' in losses:
                parts.append(f"coarse_cs={losses['coarse_cs'].item():.4f}")
            if 'coarse_grad' in losses and loss_cfg.get('lambda_coarse_grad', 0.0) > 0:
                parts.append(f"coarse_grad={losses['coarse_grad'].item():.4f}")
            if 'fine_nce' in losses:
                parts.append(f"fine_nce={losses['fine_nce'].item():.3f}")
            if 'coarse_nce' in losses:
                parts.append(f"coarse_nce={losses['coarse_nce'].item():.3f}")
            if 'compress_total' in losses:
                parts.append(f"comp={losses['compress_total'].item():.3f}")
            if 'tv' in losses:
                parts.append(f"TV={losses['tv']:.4f}")
            if 'coarse_tv' in losses:
                parts.append(f"CTV={losses['coarse_tv']:.4f}")
            if 'normal' in losses:
                parts.append(f"Nor={losses['normal']:.4f}")
            if 'distort' in losses:
                parts.append(f"Dist={losses['distort']:.4f}")

            parts.append(f"B={batch_size}")
            parts.append(f"N={gaussians.num_points:,}")
            parts.append(f"({it_s:.1f} it/s, {img_s:.1f} img/s)")

            msg = ' | '.join(parts)
            if is_best:
                msg += ' ★ BEST'
            log(msg)

        # ── Save best independently (only after coarse starts) ──
        if is_main_process and is_best and iteration >= coarse_save_start:
            best_ckpt = {
                'iteration': iteration,
                'config': cfg,
                'hash_grid_state': hash_grid.state_dict(),
                'fine_decoder_state': renderer.fine_decoder.state_dict(),
                'latent': gaussians._latent.detach().cpu(),
                'gaussians_optimizer_state': gaussians.optimizer.state_dict(),
                'gaussians_state': {
                    '_xyz': gaussians._xyz.detach().cpu(),
                    '_rotation': gaussians._rotation.detach().cpu(),
                    '_scaling': gaussians._scaling.detach().cpu(),
                    '_opacity': gaussians._opacity.detach().cpu(),
                    '_features_dc': gaussians._features_dc.detach().cpu(),
                    '_features_rest': gaussians._features_rest.detach().cpu(),
                    'active_sh_degree': gaussians.active_sh_degree,
                },
                'dcff_optimizer_state': dcff_optimizer.state_dict(),
                'dcff_scheduler_state': dcff_scheduler.state_dict(),
                'best_metrics': best_metrics,
            }
            if renderer.coarse_carrier_fusion is not None:
                best_ckpt['coarse_fusion_state'] = renderer.coarse_carrier_fusion.state_dict()
            if feat_sharp_fine is not None:
                best_ckpt['feat_sharp_fine_state'] = feat_sharp_fine.state_dict()
            if feat_select is not None:
                best_ckpt['fsm_state'] = feat_select.state_dict()
            if teacher is not None:
                best_ckpt['projection_state'] = _teacher_projection_state(teacher)
            _atomic_save_path(
                os.path.join(ckpt_dir, 'best.pth'),
                lambda path: torch.save(best_ckpt, path),
            )
            _atomic_save_path(
                os.path.join(ckpt_dir, 'best.ply'),
                lambda path: gaussians.save_ply(path),
            )

        # ── Checkpointing ──
        if is_main_process and (iteration % save_every == 0 or iteration == total_iters):
            ckpt = {
                'iteration': iteration,
                'config': cfg,
                'hash_grid_state': hash_grid.state_dict(),
                'fine_decoder_state': renderer.fine_decoder.state_dict(),
                'latent': gaussians._latent.detach().cpu(),
                'gaussians_optimizer_state': gaussians.optimizer.state_dict(),
                'dcff_optimizer_state': dcff_optimizer.state_dict(),
                'dcff_scheduler_state': dcff_scheduler.state_dict(),
                'best_metrics': best_metrics,
            }
            if renderer.coarse_carrier_fusion is not None:
                ckpt['coarse_fusion_state'] = renderer.coarse_carrier_fusion.state_dict()
            if feat_sharp_fine is not None:
                ckpt['feat_sharp_fine_state'] = feat_sharp_fine.state_dict()
            if feat_select is not None:
                ckpt['fsm_state'] = feat_select.state_dict()
            if teacher is not None:
                ckpt['projection_state'] = _teacher_projection_state(teacher)
            _atomic_save_path(
                os.path.join(ckpt_dir, 'latest.pth'),
                lambda path: torch.save(ckpt, path),
            )
            _atomic_save_path(
                os.path.join(ckpt_dir, 'latest.ply'),
                lambda path: gaussians.save_ply(path),
            )
            log(f"  [Checkpoint] Saved at iter {iteration}")

        # ── Simple validation ──
        if val_every > 0 and iteration % val_every == 0 and test_cams:
            if is_distributed:
                dist.barrier()
            if is_main_process:
                _run_validation(
                    iteration, test_cams[:10], renderer, gaussians, hash_grid,
                    loss_fn, teacher, radio_cache, cam_to_fid, teacher_mode,
                    feat_h, feat_w, coarse_h, coarse_w, longest_edge,
                    coarse_downsample, fine_feature_dim, log,
                    feat_sharp_fine=feat_sharp_fine,
                    feat_select=feat_select,
                    teacher_input_longest_edge=teacher_input_longest_edge,
                )
            if is_distributed:
                dist.barrier()

    elapsed = time.time() - start_time
    log(f"\nTraining complete in {elapsed / 3600:.1f}h")
    log(f"Best total loss: {best_metrics['total_loss']:.4f} "
        f"at iter {best_metrics.get('iteration', '?')}")
    if log_f is not None:
        log_f.close()
    _cleanup_distributed(is_distributed)


def _run_validation(iteration, test_cams, renderer, gaussians, hash_grid,
                    loss_fn, teacher, radio_cache, cam_to_fid, teacher_mode,
                    feat_h, feat_w, coarse_h, coarse_w, longest_edge,
                    coarse_downsample, feature_dim, log,
                    feat_sharp_fine=None, feat_select=None,
                    teacher_input_longest_edge=None):
    """Quick validation on a subset of test cameras."""
    renderer.eval()
    hash_grid.eval()
    if feat_select is not None:
        feat_select.eval()

    val_losses = []
    n_valid = 0

    with torch.no_grad():
        for cam in test_cams:
            # Skip cams without features in cached mode
            if teacher_mode == 'cached' and cam.uid not in cam_to_fid:
                continue

            gt_image = load_image_tensor(cam, longest_edge)
            _, _, img_h, img_w = gt_image.shape
            viewmat = cam_to_viewmat(cam).unsqueeze(0)
            K_mat = cam_to_K(cam, img_w, img_h).unsqueeze(0)

            result = renderer(
                gaussians, viewmat=viewmat, K=K_mat,
                width=img_w, height=img_h,
                render_coarse=True,
                feature_height=feat_h, feature_width=feat_w,
            )

            if feat_sharp_fine is not None:
                depth_f = F.interpolate(
                    result['depth'], (feat_h, feat_w),
                    mode='bilinear', align_corners=False)
                alpha_f = F.interpolate(
                    result['alpha'], (feat_h, feat_w),
                    mode='bilinear', align_corners=False)
                result = dict(result)
                result['fine_features'] = feat_sharp_fine(
                    result['fine_features'], depth=depth_f, alpha=alpha_f)

            if feat_select is not None and result.get('coarse_features') is not None:
                depth_f = F.interpolate(
                    result['depth'], (feat_h, feat_w),
                    mode='bilinear', align_corners=False)
                alpha_f = F.interpolate(
                    result['alpha'], (feat_h, feat_w),
                    mode='bilinear', align_corners=False)
                coarse_f = F.interpolate(
                    result['coarse_features'], (feat_h, feat_w),
                    mode='bilinear', align_corners=False)
                result = dict(result)
                fsm_r = feat_select(
                    result['fine_features'], coarse_f, alpha_f, depth_f,
                    temperature=0.5,
                )
                result['fine_features'] = fsm_r['fine_features']
                result['coarse_features'] = fsm_r['coarse_features']

            if teacher_mode in {'online', 'online_bottleneck'} and teacher is not None:
                teacher_edge = teacher_input_longest_edge or longest_edge
                teacher_img = load_image_tensor(cam, teacher_edge)
                fine_raw, coarse_raw = teacher.extract_raw(teacher_img)
                fine_target, coarse_target = teacher.project(fine_raw, coarse_raw)
                if coarse_downsample:
                    coarse_target = F.interpolate(
                        coarse_target, (coarse_h, coarse_w),
                        mode='bilinear', align_corners=False)
            elif radio_cache is not None and cam.uid in cam_to_fid:
                fid = cam_to_fid[cam.uid]
                geo, sem = radio_cache.get(fid)
                fine_target = geo.unsqueeze(0)
                coarse_sem = sem.unsqueeze(0)
                if coarse_downsample:
                    coarse_target = F.interpolate(
                        coarse_sem, (coarse_h, coarse_w),
                        mode='bilinear', align_corners=False)
                else:
                    coarse_target = coarse_sem
            else:
                continue

            result_for_loss = dict(result)
            if coarse_downsample and 'coarse_features' in result:
                result_for_loss['coarse_features'] = F.interpolate(
                    result['coarse_features'], (coarse_h, coarse_w),
                    mode='bilinear', align_corners=False)

            losses = loss_fn.compute(
                render_result=result_for_loss, gt_rgb=gt_image,
                radio_geo=fine_target, radio_sem=coarse_target,
                hash_grid=None, phase=3,
            )
            val_losses.append(losses['total'].item())
            n_valid += 1

    if val_losses:
        avg = sum(val_losses) / len(val_losses)
        log(f"  [Val {iteration}] avg_loss={avg:.4f} ({n_valid} frames)")

    renderer.train()
    hash_grid.train()
    if feat_select is not None:
        feat_select.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_feature_field_config(args.config)
    train(cfg, resume_path=args.resume)


# ═════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()
