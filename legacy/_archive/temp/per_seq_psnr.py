import sys; sys.path.insert(0, '.')
import torch, torch.nn.functional as F, numpy as np, math, os
from collections import defaultdict
from feature_3dgs.train_2dgs_geometry import GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor

train_cams, test_cams, _, _, _ = load_scene('dataset/OldHospital')
gaussians = GaussianModel2DGS(sh_degree=3)
gaussians.load_ply('output/2dgs_models/OldHospital/v3_retrain17/point_cloud/iteration_15000/point_cloud.ply')
bg = torch.zeros(3, device='cuda')

seq_psnrs = defaultdict(list)
with torch.no_grad():
    for cam in test_cams:
        pkg = render_2dgs(gaussians, cam, bg, longest_edge=0)
        img = pkg['render'].clamp(0,1)
        rw, rh = pkg['width'], pkg['height']
        gt = load_image_tensor(cam)
        gt = F.interpolate(gt.unsqueeze(0), size=(rh,rw), mode='bilinear', align_corners=False).squeeze(0)
        mse = F.mse_loss(img, gt)
        if mse > 0:
            psnr = -10 * math.log10(mse.item())
            seq = cam.image_name.split('/')[0]
            seq_psnrs[seq].append(psnr)

print('Per-sequence PSNR:')
for seq in sorted(seq_psnrs.keys()):
    p = np.array(seq_psnrs[seq])
    print(f'  {seq}: N={len(p)}, Mean={p.mean():.2f}, Min={p.min():.2f}, Max={p.max():.2f}')

train_seqs = defaultdict(int)
for c in train_cams:
    train_seqs[c.image_name.split('/')[0]] += 1
print('\nTrain views per seq:')
for s in sorted(train_seqs.keys()):
    print(f'  {s}: {train_seqs[s]} train, {len(seq_psnrs.get(s, []))} test')
