"""Quick memory profiling for DCFF to find max batch size."""

import os
import sys
import torch
import argparse
import numpy as np
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_field.dcff import HybridGaussianModel, SpatialHashGrid, DeferredCascadedRenderer
from feature_field.utils.scene_colmap import load_scene_colmap
from feature_field.utils.project_config import load_feature_field_config


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def profile_batch(batch_size, cfg_path, longest_edge=640):
    cfg = load_feature_field_config(cfg_path)
    dcfg = cfg['dataset']
    mcfg = cfg['model']
    hcfg = cfg['hash_grid']
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    train_cams, _, pcd_xyz, pcd_rgb, cameras_extent = load_scene_colmap(
        dcfg['source_dir'], dcfg.get('images', ''))
    
    gaussians = HybridGaussianModel(sh_degree=3, latent_dim=32)
    init_ply = cfg['training'].get('init_ply', None)
    if init_ply:
        gaussians.load_ply(init_ply, freeze_geometry=False)
    else:
        gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)
    
    xyz = gaussians.get_xyz
    scene_extent = float(np.percentile(xyz.detach().cpu().numpy().reshape(-1, 3), 99, axis=0).max()) * 1.2
    
    hash_grid = SpatialHashGrid(
        scene_extent=scene_extent, feature_dim=64, input_mode='implicit_scale',
        latent_dim=32, scale_dim=2, scale_pe_freqs=4, include_raw_scale=True,
        n_levels=16, n_features_per_level=2, log2_hashmap_size=20,
        base_resolution=16, max_resolution=4096, mlp_hidden=128, mlp_layers=2,
    ).cuda()
    
    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid, latent_dim=32,
        fine_feature_dim=64, coarse_feature_dim=64,
        fine_hidden_dim=128, fine_num_layers=3, fine_use_viewdirs=False,
        fine_decoder_type='spatial',
        coarse_mode='carrier_residual',
        coarse_carrier_hidden_dim=128, coarse_gate_hidden_dim=64,
    ).cuda()
    
    args = type('Args', (), {
        'position_lr_init': 0.00016, 'position_lr_final': 0.0000016,
        'feature_lr': 0.0025, 'opacity_lr': 0.05, 'scaling_lr': 0.005,
        'rotation_lr': 0.001, 'latent_lr': 0.0003,
        'percent_dense': 0.01, 'iterations': 50000,
    })()
    gaussians.training_setup(args)
    
    dcff_params = [
        {'params': hash_grid.parameters(), 'lr': 0.0002},
        {'params': renderer.fine_decoder.parameters(), 'lr': 0.0006},
    ]
    if renderer.coarse_carrier_fusion is not None:
        dcff_params.append({'params': renderer.coarse_carrier_fusion.parameters(), 'lr': 0.0006})
    dcff_opt = torch.optim.Adam(dcff_params, eps=1e-15)
    
    print(f"\n  Param counts:")
    print(f"    Gaussians: {gaussians.num_points:,}")
    print(f"    Hash grid: {count_params(hash_grid):,}")
    print(f"    Fine dec:  {count_params(renderer.fine_decoder):,}")
    if renderer.coarse_carrier_fusion:
        print(f"    Coarse f:  {count_params(renderer.coarse_carrier_fusion):,}")
    
    cam = train_cams[0]
    from PIL import Image
    from torchvision import transforms
    import math
    
    img = Image.open(cam.image).convert('RGB')
    W, H = img.size
    scale = longest_edge / max(W, H)
    if scale < 1.0:
        img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()
    _, _, H_r, W_r = img_tensor.shape
    
    images = img_tensor.expand(batch_size, -1, -1, -1)
    
    def cam_to_viewmat(c):
        W2C = np.eye(4)
        W2C[:3, :3] = c.R.T
        W2C[:3, 3] = c.T
        return torch.tensor(W2C, dtype=torch.float32, device="cuda")
    
    def cam_to_K(c, w, h):
        tanfovx = math.tan(c.FovX * 0.5)
        tanfovy = math.tan(c.FovY * 0.5)
        fx = w / (2 * tanfovx)
        fy = h / (2 * tanfovy)
        return torch.tensor([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=torch.float32, device="cuda")
    
    viewmats = torch.stack([cam_to_viewmat(cam) for _ in range(batch_size)], dim=0)
    Ks = torch.stack([cam_to_K(cam, W_r, H_r) for _ in range(batch_size)], dim=0)
    
    feat_h = H_r // 4
    feat_w = W_r // 4
    
    try:
        result = renderer(
            gaussians, viewmat=viewmats, K=Ks,
            width=W_r, height=H_r,
            render_coarse=True,
            feature_height=feat_h, feature_width=feat_w,
        )
        
        loss = result['rgb'].mean() + result['fine_features'].mean() + result['coarse_features'].mean()
        
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  Batch={batch_size}: Peak={peak_mem:.1f}GB | OK")
        
        del result, loss, images, viewmats, Ks
        del hash_grid, renderer, gaussians, dcff_opt
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        return peak_mem
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n  Batch={batch_size}: OOM at {peak_mem:.1f}GB | FAIL")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return None
        raise


def main():
    cfg_path = 'feature_field/configs/dcff_oldhospital_v10c_carrier_residual.yaml'
    print("="*60)
    print("DCFF Memory Profiling")
    print("="*60)
    
    for bs in [8, 12, 16, 20, 24, 32]:
        result = profile_batch(bs, cfg_path)
        if result is None:
            print(f"  -> Max batch = {bs-4}")
            break
    
    print("\nDone.")


if __name__ == '__main__':
    main()