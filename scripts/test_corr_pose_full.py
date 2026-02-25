#!/usr/bin/env python3
"""Test CorrPoseNet full forward pass with actual renderer + dataset."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import time

from ic_models.corr_pose_net import CorrPoseNet
from data.dataset_v3 import PoseDatasetV3
from modules.multiscale_renderer import MultiScaleRenderer

device = torch.device('cuda')
INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}

# --- Load renderer ---
print("Loading renderer...")
renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
    img_height=480, img_width=640,
    fx=320.0, fy=320.0, cx=319.5, cy=239.5,
)
print("  Renderer loaded")

# --- Load dataset (single sample) ---
dataset = PoseDatasetV3(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir='dataset/room_0/Sequence_1/depth',
    frame_indices=list(range(10)),
    scale_names=['fine_dino'],
    noise_rot_deg=10.0,
    noise_trans_m=0.2,
    is_train=True,
    depth_resize=(35, 46),
)

sample = dataset[0]
query = sample['query_feats']['fine_dino'].unsqueeze(0).to(device)
pose_gt = sample['pose_gt'].unsqueeze(0).to(device)
initial_pose = sample['initial_pose'].unsqueeze(0).to(device)
depth = sample['depth'].unsqueeze(0).to(device)

print(f"  Query shape: {query.shape}")
print(f"  Initial pose shape: {initial_pose.shape}")
print(f"  Depth shape: {depth.shape}")

# --- Init network ---
net = CorrPoseNet(feat_dim=768, enc_dim=128, hidden_dim=128, corr_radius=4, num_iters=3).to(device)
print(f"\n{net}")

# --- Forward pass ---
print("\nRunning forward pass (3 iterations with rendering)...")
t0 = time.time()

# Test with training mode (gradients)
net.train()
results = net(
    query_feats=query,
    initial_pose=initial_pose,
    depth=depth,
    intrinsics=INTRINSICS,
    renderer=renderer,
    scale_name='fine_dino',
)
elapsed = time.time() - t0
print(f"  Forward time: {elapsed:.2f}s")
print(f"  Poses: {len(results['poses'])} (initial + {len(results['flows'])} iterations)")
print(f"  Flows: {[f.shape for f in results['flows']]}")
print(f"  Confidences: {[c.shape for c in results['confidences']]}")
print(f"  Delta_xis: {[d.shape for d in results['delta_xis']]}")

# --- Compute errors ---
from scripts.train_corr_pose import compute_pose_error
init_rot, init_trans = compute_pose_error(initial_pose, pose_gt)
final_rot, final_trans = compute_pose_error(results['poses'][-1], pose_gt)
print(f"\n  Init error:  rot={init_rot.item():.2f}°  trans={init_trans.item():.4f}m")
print(f"  Final error: rot={final_rot.item():.2f}°  trans={final_trans.item():.4f}m")

# --- Test backward ---
print("\nTesting backward pass...")
from scripts.train_corr_pose import pose_loss_fn
from modules.lie_algebra import pose_inverse
loss = pose_loss_fn(results['poses'], pose_gt, gamma=0.8)
print(f"  Loss: {loss.item():.4f}")

t0 = time.time()
loss.backward()
backward_time = time.time() - t0
print(f"  Backward time: {backward_time:.2f}s")

# Check gradients on all modules
for name, param in net.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 0:
            print(f"  {name}: grad_norm={grad_norm:.6f}")

print(f"\n=== Full forward+backward test passed! ===")
print(f"Total time: {elapsed + backward_time:.2f}s for 3 iterations")
print(f"Estimated training: {(elapsed + backward_time) * 810 / 60:.0f} min/epoch")
