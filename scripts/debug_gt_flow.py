#!/usr/bin/env python3
"""Debug GT flow computation to find source of huge EPE values."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ic_models.ms_flow_pose_net import MSFlowPoseNet
from data.dataset_v4 import PoseDatasetV4, collate_v4
from torch.utils.data import DataLoader

device = 'cuda:0'

# Load dataset
ds = PoseDatasetV4(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir='dataset/room_0/Sequence_1/depth',
    noise_rot_deg=8.0, noise_trans_m=0.25, is_train=True)
dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_v4)

model = MSFlowPoseNet(
    coarse_hw=(7, 10), mid_hw=(15, 20), fine_hw=(35, 46)).to(device)

for i, batch in enumerate(dl):
    if i >= 5:
        break
    
    pose_init = batch['initial_pose'].to(device)
    pose_gt = batch['pose_gt'].to(device)
    
    # Get depth from dataset or create dummy
    if 'depth' in batch and batch['depth'] is not None:
        depth = batch['depth'].to(device)
    else:
        depth = torch.ones(pose_init.shape[0], 35, 46, device=device) * 2.0
    
    print(f"\n=== Batch {i} ===")
    print(f"Depth: shape={depth.shape}, min={depth.min():.3f}, max={depth.max():.3f}, "
          f"mean={depth.mean():.3f}, zeros={(depth < 0.01).sum().item()}/{depth.numel()}")
    
    # Relative pose
    T_rel = pose_gt @ torch.linalg.inv(pose_init)
    trans_norm = T_rel[:, :3, 3].norm(dim=1)
    R_rel = T_rel[:, :3, :3]
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)) * 180 / math.pi
    print(f"Relative pose: rot={angle.tolist()}, trans(m)={trans_norm.tolist()}")
    
    # GT flow (now returns tuple)
    with torch.no_grad():
        gt_fine, mask_fine = model.compute_gt_flow(pose_init, pose_gt, depth, (35, 46))
    
    valid_ratio = mask_fine.mean().item()
    epe = torch.norm(gt_fine, dim=1)  # includes masked-out zeros
    # Masked EPE (only valid pixels)
    epe_map = torch.norm(gt_fine, dim=1, keepdim=True)
    n_valid = mask_fine.sum().clamp(min=1)
    masked_epe = (epe_map * mask_fine).sum() / n_valid
    print(f"GT flow fine: du=[{gt_fine[:, 0].min():.1f}, {gt_fine[:, 0].max():.1f}], "
          f"dv=[{gt_fine[:, 1].min():.1f}, {gt_fine[:, 1].max():.1f}]")
    print(f"  Valid pixels: {valid_ratio*100:.1f}%")
    print(f"  All EPE: min={epe.min():.1f}, max={epe.max():.1f}, mean={epe.mean():.1f}")
    print(f"  Masked EPE (valid only): {masked_epe.item():.1f}")
    print(f"  >50px: {(epe > 50).sum().item()}/{epe.numel()} ({(epe > 50).float().mean()*100:.1f}%)")
    print(f"  >100px: {(epe > 100).sum().item()}, >1000px: {(epe > 1000).sum().item()}")

    # Also check coarse GT flow
    gt_coarse, mask_coarse = model.compute_gt_flow(pose_init, pose_gt, depth, (7, 10))
    epe_c = torch.norm(gt_coarse, dim=1)
    print(f"GT flow coarse: EPE mean={epe_c.mean():.1f}, max={epe_c.max():.1f}, valid={mask_coarse.mean()*100:.1f}%")

print("\n=== Summary ===")
print(f"Image size: 35x46, diagonal= ~{(35**2 + 46**2)**0.5:.0f}px")
print("If max EPE >> diagonal, GT flow computation has a bug!")
