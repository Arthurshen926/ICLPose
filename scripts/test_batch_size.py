#!/usr/bin/env python3
"""测试不同 batch_size 下的显存占用和训练速度."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import time
import gc

from ic_models.corr_pose_net import CorrPoseNet
from modules.multiscale_renderer import MultiScaleRenderer
from data.dataset_v3 import PoseDatasetV3
from modules.lie_algebra import se3_exp

INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

device = torch.device('cuda')

# Load renderer
print("Loading renderer...")
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth'},
    device=device, img_height=480, img_width=640,
    fx=320.0, fy=320.0, cx=319.5, cy=239.5,
)

# Load dataset
dataset = PoseDatasetV3(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir='dataset/room_0/Sequence_1/depth',
    frame_indices=list(range(810)),
    scale_names=['fine_dino'],
    noise_rot_deg=10.0, noise_trans_m=0.2,
    is_train=True, depth_resize=(35, 46),
)

# Network
net = CorrPoseNet(feat_dim=768, num_iters=3).to(device)

def get_mem():
    return torch.cuda.max_memory_allocated() / 1024**3

def test_batch(bs, num_batches=5):
    """Test a specific batch size."""
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4)
    net.train()
    
    times = []
    for b in range(num_batches):
        # Collect batch
        samples = [dataset[i] for i in range(b * bs, b * bs + bs)]
        queries = torch.stack([s['query_feats']['fine_dino'] for s in samples]).to(device)
        poses_gt = torch.stack([s['pose_gt'] for s in samples]).to(device)
        initials = torch.stack([s['initial_pose'] for s in samples]).to(device)
        depths = torch.stack([s['depth'] for s in samples]).to(device)
        
        torch.cuda.synchronize()
        t0 = time.time()
        
        results = net(
            query_feats=queries,
            initial_pose=initials,
            depth=depths,
            intrinsics=INTRINSICS,
            renderer=renderer,
            scale_name='fine_dino',
        )
        
        # Simple loss
        R_pred = results['poses'][-1][:, :3, :3]
        R_gt = poses_gt[:, :3, :3]
        R_rel = R_pred @ R_gt.transpose(-1, -2)
        cos_a = (R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        loss = (1 - cos_a.clamp(-1, 1)).mean()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        t1 = time.time()
        
        times.append(t1 - t0)
    
    peak_mem = get_mem()
    avg_time = sum(times[1:]) / len(times[1:])  # skip warmup
    samples_per_sec = bs / avg_time
    
    return peak_mem, avg_time, samples_per_sec

print(f"\nBaseline GPU memory after model load: {get_mem():.2f} GB")
print(f"{'BS':>4} {'Peak GB':>9} {'Time/batch':>12} {'Samples/s':>11} {'Speedup':>9}")
print("-" * 55)

ref_sps = None
for bs in [1, 2, 4]:
    try:
        peak, avg_t, sps = test_batch(bs, num_batches=4)
        if ref_sps is None:
            ref_sps = sps
        speedup = sps / ref_sps
        print(f"{bs:4d} {peak:8.2f}G {avg_t:11.2f}s {sps:10.2f}/s {speedup:8.2f}x")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"{bs:4d}   OOM!")
            torch.cuda.empty_cache()
            gc.collect()
            break
        else:
            raise
