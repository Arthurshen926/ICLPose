#!/usr/bin/env python3
"""Quick 1-batch training sanity check (no renderer, uses query feats as render feats)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from data.dataset_v4 import PoseDatasetV4, collate_v4
from torch.utils.data import DataLoader
from ic_models.ms_flow_pose_net import MSFlowPoseNet
from modules.lie_algebra import se3_exp

device = 'cuda'

# 1. Dataset
ds = PoseDatasetV4(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir='dataset/room_0/Sequence_1/depth',
    noise_rot_deg=15.0, noise_trans_m=0.5, is_train=True,
)
dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_v4, num_workers=0)
batch = next(iter(dl))

# 2. Model
model = MSFlowPoseNet(
    coarse_hw=(7, 10), mid_hw=(15, 20), fine_hw=(35, 46),
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 3. Feed (use query_feats as render_feats for quick sanity check)
query_feats = {k: v.to(device) for k, v in batch['query_feats'].items()}
pose_gt = batch['pose_gt'].to(device)
pose_init = batch['initial_pose'].to(device)
depth = batch['depth'].to(device)

# Simulate render_feats from a slightly different viewpoint
render_feats = {k: v + torch.randn_like(v) * 0.01 for k, v in query_feats.items()}

# 4. Forward
model.train()
out = model(query_feats, render_feats, depth)

# 5. GT flows
gt_flows = {}
for name, hw in [('coarse', model.COARSE_HW), ('mid', model.MID_HW), ('fine', model.FINE_HW)]:
    gt_flows[name] = model.compute_gt_flow(pose_init, pose_gt, depth, hw)

# 6. Flow loss
total_loss = torch.tensor(0.0, device=device)
weights = {'coarse': 0.1, 'mid': 0.3, 'fine': 1.0}
scale_map = {'coarse': 'flow_coarse', 'mid': 'flow_mid', 'fine': 'flow_fine'}
for scale_name, pred_key in scale_map.items():
    l1 = F.l1_loss(out[pred_key], gt_flows[scale_name])
    epe = torch.norm(out[pred_key] - gt_flows[scale_name], dim=1).mean()
    total_loss = total_loss + weights[scale_name] * l1
    print(f"  {scale_name:8s}: L1={l1.item():.4f}, EPE={epe.item():.4f}")

# 7. Pose loss
if 'delta_xi' in out:
    import math
    T_delta = se3_exp(out['delta_xi'])
    pose_pred = torch.bmm(T_delta, pose_init)
    R_pred, R_gt = pose_pred[:, :3, :3], pose_gt[:, :3, :3]
    R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1 + 1e-7, 1 - 1e-7)
    rot_err = torch.acos(cos_angle) * 180.0 / math.pi
    trans_err = torch.norm(pose_pred[:, :3, 3] - pose_gt[:, :3, 3], dim=1) * 1000
    print(f"\n  Rot err: {rot_err.mean().item():.2f}° (median {rot_err.median().item():.2f}°)")
    print(f"  Trans err: {trans_err.mean().item():.1f}mm")

# 8. Backward
print(f"\n  Total flow loss: {total_loss.item():.4f}")
optimizer.zero_grad()
total_loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print(f"  Grad norm: {grad_norm.item():.4f}")

print("\n✓ 1-batch training sanity check PASSED!")
