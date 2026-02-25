#!/usr/bin/env python3
"""快速benchmark: 不同batch_size下CorrPoseNet forward的速度和显存"""
import torch
import time
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

from modules.multiscale_renderer import MultiScaleRenderer
from ic_models.corr_pose_net import CorrPoseNet

def main():
    device = torch.device('cuda:0')
    
    # Load renderer
    print('Loading renderer...')
    ply_path = 'dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply'
    scale_paths = {'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth'}
    renderer = MultiScaleRenderer(ply_path=ply_path, scale_model_paths=scale_paths, device='cuda:0')
    
    feat_dim = renderer.scale_info['fine_dino']['feat_dim']  # 768
    fH, fW = renderer.scale_info['fine_dino']['resolution']  # 35, 46
    
    INTRINSICS = {
        'fx': 320.0 * (fW / 640), 'fy': 320.0 * (fH / 480),
        'cx': 319.5 * (fW / 640), 'cy': 239.5 * (fH / 480),
    }
    
    # Create network
    net = CorrPoseNet(feat_dim=feat_dim, enc_dim=128, hidden_dim=128, 
                      corr_radius=4, num_iters=3, damping=1e-3).to(device)
    net.train()
    
    print(f'Feature: {feat_dim}d, {fH}x{fW}')
    print()
    
    for BS in [1, 2, 4, 8]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create dummy data
        query = torch.randn(BS, feat_dim, fH, fW, device=device)
        depth = torch.ones(BS, fH, fW, device=device) * 1.5
        pose_gt = torch.eye(4, device=device).unsqueeze(0).expand(BS, -1, -1).clone()
        initial_pose = pose_gt.clone()
        # Add noise
        initial_pose[:, :3, 3] += torch.randn(BS, 3, device=device) * 0.1
        
        # Warmup
        try:
            results = net(query, initial_pose, depth, INTRINSICS, renderer, 'fine_dino')
            loss = sum(((p[:, :3, 3] - pose_gt[:, :3, 3])**2).sum() for p in results['poses'][1:])
            loss.backward()
            net.zero_grad()
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'BS={BS}: OOM or error: {e}')
            break
        
        # Benchmark
        times = []
        for trial in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            results = net(query, initial_pose, depth, INTRINSICS, renderer, 'fine_dino')
            loss = sum(((p[:, :3, 3] - pose_gt[:, :3, 3])**2).sum() for p in results['poses'][1:])
            loss.backward()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            net.zero_grad()
        
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3
        avg_time = sum(times) / len(times)
        per_sample = avg_time / BS
        
        print(f'BS={BS}: total={avg_time*1000:.0f}ms, per_sample={per_sample*1000:.0f}ms, '
              f'peak_mem={mem_peak:.1f}GB, throughput={BS/avg_time:.2f} samples/s')
    
    print('\nDone!')

if __name__ == '__main__':
    main()
