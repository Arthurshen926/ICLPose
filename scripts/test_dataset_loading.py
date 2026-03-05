#!/usr/bin/env python3
"""Quick test: load v1 features via DatasetV4 and verify shapes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_v4 import PoseDatasetV4, collate_v4
from torch.utils.data import DataLoader

ds = PoseDatasetV4(
    feature_base_dir='output/features_multiscale/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    depth_dir='dataset/room_0/Sequence_1/depth',
    noise_rot_deg=15.0,
    noise_trans_m=0.5,
    is_train=True,
)
print(f'Dataset size: {len(ds)}')

sample = ds[0]
print('Keys:', list(sample.keys()))
for k, v in sample['query_feats'].items():
    print(f'  {k}: {v.shape}')
if 'depth' in sample:
    print(f'  depth: {sample["depth"].shape}')
print(f'  pose_gt: {sample["pose_gt"].shape}')
print(f'  initial_pose: {sample["initial_pose"].shape}')

# Test collation
dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_v4, num_workers=0)
batch = next(iter(dl))
print('\nBatch:')
for k, v in batch['query_feats'].items():
    print(f'  {k}: {v.shape}')
print(f'  depth: {batch["depth"].shape}')
print(f'  pose_gt: {batch["pose_gt"].shape}')

# End-to-end: feed batch to model
import torch
from ic_models.ms_flow_pose_net import MSFlowPoseNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MSFlowPoseNet(
    coarse_hw=(7, 10), mid_hw=(15, 20), fine_hw=(35, 46),
).to(device)

query_feats = {k: v.to(device) for k, v in batch['query_feats'].items()}
depth = batch['depth'].to(device)

# Simulate render_feats = query_feats (for testing only)
render_feats = query_feats

with torch.no_grad():
    out = model(query_feats, render_feats, depth)

print('\nModel outputs:')
for k, v in out.items():
    if isinstance(v, torch.Tensor):
        print(f'  {k}: {v.shape}')

print('\n✓ End-to-end dataset → model test PASSED!')
