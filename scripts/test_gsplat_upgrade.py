#!/usr/bin/env python3
"""测试 gsplat v1.5+ 升级后的渲染功能"""

import torch
import time
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

from modules.multiscale_renderer import MultiScaleRenderer

def main():
    ply_path = 'dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply'
    scale_paths = {
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    }

    print('Loading renderer...')
    renderer = MultiScaleRenderer(
        ply_path=ply_path,
        scale_model_paths=scale_paths,
        device='cuda:0',
    )
    print()

    # 单帧渲染测试
    pose = torch.eye(4, device='cuda:0')
    print('=== Single frame render ===')
    # warmup
    _ = renderer.render_scale('fine_dino', pose)
    torch.cuda.synchronize()
    
    t0 = time.time()
    result = renderer.render_scale('fine_dino', pose)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f'feature_map shape: {result["feature_map"].shape}')
    print(f'Time: {(t1-t0)*1000:.1f}ms')
    print()

    # 批量渲染测试 (B=4)
    poses = torch.eye(4, device='cuda:0').unsqueeze(0).expand(4, -1, -1).clone()
    poses[:, :3, 3] += torch.randn(4, 3, device='cuda:0') * 0.1

    print('=== Batch render (B=4) ===')
    t0 = time.time()
    batch_result = renderer.render_batch(poses, scales=['fine_dino'], return_depth=True)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f'feat shape: {batch_result["fine_dino_feat"].shape}')
    print(f'depth shape: {batch_result["depth_map"].shape}')
    print(f'Time: {(t1-t0)*1000:.1f}ms')
    print(f'Per-frame: {(t1-t0)*1000/4:.1f}ms')
    print()

    # 深度渲染测试
    print('=== Depth render ===')
    t0 = time.time()
    result_d = renderer.render_scale('fine_dino', pose, return_depth=True)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f'depth shape: {result_d["depth_map"].shape}')
    print(f'depth range: [{result_d["depth_map"].min():.2f}, {result_d["depth_map"].max():.2f}]')
    print(f'Time: {(t1-t0)*1000:.1f}ms')

    # 批量B=1 vs B=4 vs B=8 benchmark
    print()
    print('=== Batch rendering benchmark ===')
    for B in [1, 2, 4, 8]:
        poses = torch.eye(4, device='cuda:0').unsqueeze(0).expand(B, -1, -1).clone()
        poses[:, :3, 3] += torch.randn(B, 3, device='cuda:0') * 0.1
        
        # warmup
        _ = renderer.render_batch(poses, scales=['fine_dino'], return_depth=False)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(5):
            t0 = time.time()
            _ = renderer.render_batch(poses, scales=['fine_dino'], return_depth=False)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        
        avg = sum(times) / len(times)
        print(f'  B={B}: total={avg*1000:.1f}ms, per-frame={avg*1000/B:.1f}ms')

    print()
    print('ALL TESTS PASSED!')


if __name__ == '__main__':
    main()
