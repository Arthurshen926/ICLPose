#!/usr/bin/env python3
"""
Quick benchmark: Compare position-NN vs DINOv2 CLS-token retrieval for initial pose.
Reports median position/rotation error of the nearest training pose for each method.
"""
import json, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

device = torch.device('cuda')
source_dir = Path('dataset/OldHospital')
cameras_json = 'output/2dgs_models/OldHospital/v7_depth/cameras.json'
render_w, render_h = 960, 540

with open(cameras_json) as f:
    all_cams = json.load(f)
cam_by_name = {c['img_name']: c for c in all_cams}

# Load splits
def load_split(name):
    samples = []
    with open(source_dir / f'dataset_{name}.txt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('V') or line.startswith('I'):
                continue
            samples.append(line.split()[0])
    return samples

train_names = load_split('train')
test_names = load_split('test')
train_cams = [cam_by_name[n] for n in train_names if n in cam_by_name]
train_positions = np.array([c['position'] for c in train_cams])

# Load DINOv2
print("Loading DINOv2...")
backbone = torch.hub.load('/root/.cache/torch/hub/facebookresearch_dinov2_main', 
                          'dinov2_vitb14', source='local').cuda().eval()

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def extract_cls(img_path):
    img = Image.open(img_path).convert('RGB').resize((render_w, render_h), Image.BILINEAR)
    img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_norm = (img_t.unsqueeze(0).to(device) - MEAN) / STD
    ps = 14
    H, W = img_norm.shape[2], img_norm.shape[3]
    pad_h = (ps - H % ps) % ps
    pad_w = (ps - W % ps) % ps
    if pad_h > 0 or pad_w > 0:
        img_norm = F.pad(img_norm, (0, pad_w, 0, pad_h), mode='reflect')
    with torch.no_grad():
        out = backbone.forward_features(img_norm)
    return out['x_norm_clstoken']  # [1, 768]

# Build train DB
print("Building train DB...")
train_descs = []
for cam in tqdm(train_cams):
    p = source_dir / cam['img_name']
    if not p.exists():
        p = source_dir / 'processed' / cam['img_name']
    train_descs.append(extract_cls(p).cpu())
train_descs = F.normalize(torch.cat(train_descs, dim=0), p=2, dim=1)  # [N, 768]

def pose_error(pred_cam, gt_cam):
    pred_c2w = np.eye(4, dtype=np.float32)
    pred_c2w[:3, :3] = np.array(pred_cam['rotation'], dtype=np.float32)
    pred_c2w[:3, 3] = np.array(pred_cam['position'], dtype=np.float32)
    gt_c2w = np.eye(4, dtype=np.float32)
    gt_c2w[:3, :3] = np.array(gt_cam['rotation'], dtype=np.float32)
    gt_c2w[:3, 3] = np.array(gt_cam['position'], dtype=np.float32)
    pos_err = np.linalg.norm(pred_c2w[:3, 3] - gt_c2w[:3, 3]) * 100
    R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
    trace = np.clip(np.trace(R_rel), -1, 3)
    rot_err = math.acos(np.clip((trace - 1) / 2, -1, 1)) * 180 / math.pi
    return pos_err, rot_err

# Evaluate both methods
print("\nEvaluating...")
pos_nn, rot_nn, pos_dino, rot_dino = [], [], [], []

for name in tqdm(test_names):
    if name not in cam_by_name:
        continue
    gt_cam = cam_by_name[name]
    gt_pos = np.array(gt_cam['position'])
    
    # Position-NN
    dists = np.linalg.norm(train_positions - gt_pos[None, :], axis=1)
    nn_cam = train_cams[np.argmin(dists)]
    pe, re = pose_error(nn_cam, gt_cam)
    pos_nn.append(pe); rot_nn.append(re)
    
    # DINOv2 retrieval
    p = source_dir / name
    if not p.exists():
        p = source_dir / 'processed' / name
    q_desc = F.normalize(extract_cls(p).cpu(), p=2, dim=1)
    sims = (q_desc @ train_descs.T).squeeze(0)
    best_idx = sims.argmax().item()
    dino_cam = train_cams[best_idx]
    pe2, re2 = pose_error(dino_cam, gt_cam)
    pos_dino.append(pe2); rot_dino.append(re2)

pos_nn, rot_nn = np.array(pos_nn), np.array(rot_nn)
pos_dino, rot_dino = np.array(pos_dino), np.array(rot_dino)

print(f"\n{'='*60}")
print(f"Position-NN Retrieval:")
print(f"  Median: {np.median(pos_nn):.1f} cm / {np.median(rot_nn):.2f}°")
print(f"  Mean:   {np.mean(pos_nn):.1f} cm / {np.mean(rot_nn):.2f}°")
print(f"  P90:    {np.percentile(pos_nn, 90):.1f} cm / {np.percentile(rot_nn, 90):.2f}°")
print(f"\nDINOv2 CLS-Token Retrieval:")
print(f"  Median: {np.median(pos_dino):.1f} cm / {np.median(rot_dino):.2f}°")
print(f"  Mean:   {np.mean(pos_dino):.1f} cm / {np.mean(rot_dino):.2f}°")
print(f"  P90:    {np.percentile(pos_dino, 90):.1f} cm / {np.percentile(rot_dino, 90):.2f}°")
print(f"{'='*60}")

# Per-sample comparison
n_better_pos = (pos_dino < pos_nn).sum()
n_better_rot = (rot_dino < rot_nn).sum()
print(f"\nDINOv2 better on {n_better_pos}/{len(pos_nn)} position, {n_better_rot}/{len(rot_nn)} rotation")
