"""Diagnose retrain7 model quality issues."""
import torch, json, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.visualize_2dgs_recon import load_ply_2dgs, cam_to_viewmat, render_2dgs

model_dir = 'output/2dgs_models/OldHospital/v3_retrain7'
ply_path = f'{model_dir}/point_cloud/iteration_15000/point_cloud.ply'
model = load_ply_2dgs(ply_path)
n = model['xyz'].shape[0]

# Scale distribution
scales = torch.exp(model['scales'])
max_s = scales.max(dim=1).values
print(f'=== Gaussians: {n:,} ===')
print(f'Scale (exp): min={max_s.min():.6f}, max={max_s.max():.4f}, mean={max_s.mean():.6f}, median={max_s.median():.6f}')
for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
    print(f'  Scale > {t}: {(max_s > t).sum().item():,}')

# Opacity distribution
opacities = torch.sigmoid(model['opacities'])
print(f'\nOpacity: min={opacities.min():.4f}, max={opacities.max():.4f}, mean={opacities.mean():.4f}')
for t in [0.01, 0.1, 0.5]:
    print(f'  Opacity < {t}: {(opacities < t).sum().item():,}')

# Render test view
cams = json.load(open(f'{model_dir}/cameras.json'))
test_cam = [c for c in cams if 'seq8/frame00051' in c['img_name']][0]
w, h = test_cam['width'], test_cam['height']
fx, fy = test_cam['fx'], test_cam['fy']
viewmat = torch.tensor(cam_to_viewmat(test_cam), device='cuda')
K = torch.tensor([[fx, 0, w/2.0], [0, fy, h/2.0], [0, 0, 1]], device='cuda')
with torch.no_grad():
    rgb, depth, alpha, normal = render_2dgs(model, viewmat, K, w, h)
print(f'\nRendered depth: min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}')
print(f'Rendered alpha: min={alpha.min():.4f}, max={alpha.max():.4f}, mean={alpha.mean():.4f}')

# Mono depth correlation
d = np.load('dataset/OldHospital/mono_depth/seq8/frame00051.npy')
mono = 1.0 - torch.from_numpy(d).float().cuda()
rd = depth.squeeze()
md = mono
valid = (rd > 0.01) & (md > 0.001)
rd_v = rd[valid]
md_v = md[valid]
pearson = torch.corrcoef(torch.stack([rd_v, md_v]))[0, 1]
print(f'\nPearson correlation (rendered vs mono): {pearson:.4f}')
print(f'  (should be close to 1.0 if depth direction is correct)')

# Also check inverse direction
mono_inv = torch.from_numpy(d).float().cuda()
md_inv_v = mono_inv[valid]
pearson_inv = torch.corrcoef(torch.stack([rd_v, md_inv_v]))[0, 1]
print(f'Pearson correlation (rendered vs raw_mono): {pearson_inv:.4f}')
print(f'  (negative means inversion is correct)')
