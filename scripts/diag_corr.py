#!/usr/bin/env python3
"""Quick diagnostic: correlation volume quality at GT pose."""
import sys, os, math, yaml, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ic_models.radio_pose_net import RadioPoseNet, global_correlation
from scripts.train_radio_pose import RadioRenderer, RadioPoseDataset, collate_radio
import torch.nn.functional as F

torch.set_grad_enabled(False)
cfg = yaml.safe_load(open('configs/radio_pose_oh_v12.yaml'))
rc, mc, dc = cfg['renderer'], cfg['model'], cfg['data']
device = torch.device('cuda')

renderer = RadioRenderer(rc['ply_path'], rc['feature_model_path'], device,
    tuple(rc['img_hw']), rc['fx'], rc['fy'], rc['cx'], rc['cy'])
intrinsics = {k: rc[k] for k in ('fx', 'fy', 'cx', 'cy')}
model = RadioPoseNet(
    feat_dim=64, hidden_dim=128, n_heads=4, n_attn_layers=2, ffn_dim=128,
    local_radius=4, fine_iters=4, damping=0.001, coarse_hw=(17,30), fine_hw=(34,60),
    intrinsics=intrinsics, img_hw=(1080,1920), conf_floor=0.1,
    detach_conf_in_solver=True, solver_hw=(68,120), sequential_solve=True,
    solver_trans_scale=0.0,
).to(device).eval()
ckpt = torch.load('output/radio_pose_oh_v12/checkpoints/latest.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f"Loaded E{ckpt.get('epoch', '?')}")

test_indices = np.load(os.path.join(dc['feature_dir'], 'test_indices.npy')).tolist()

# Near-GT pose test
torch.manual_seed(42); np.random.seed(42)
val_ds = RadioPoseDataset(dc['feature_dir'], dc['traj_path'],
    frame_indices=test_indices[:4], noise_rot_deg=0.01, noise_trans_m=0.001, is_train=True)
batch = next(iter(torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_radio)))

q = batch['query_feat'].to(device)
gt = batch['pose_gt'].to(device)
rf = renderer.render_features(gt, (68, 120))

q_n = F.normalize(q, dim=1)
r_n = F.normalize(rf, dim=1)

# Raw cosine similarity (per pixel, same position)
cos_raw = F.cosine_similarity(
    F.interpolate(q_n, (17,30), mode='bilinear', align_corners=False),
    F.interpolate(r_n, (17,30), mode='bilinear', align_corners=False), dim=1)
print(f"Raw cosine sim @ GT: mean={cos_raw.mean():.3f}")

# After domain projection
q_c = F.interpolate(q_n, (17,30), mode='bilinear', align_corners=False)
r_c = F.interpolate(r_n, (17,30), mode='bilinear', align_corners=False)
q_p = model.proj_q(q_c)
r_p = model.proj_r(r_c)
cos_proj = F.cosine_similarity(q_p, r_p, dim=1)
print(f"Projected cosine sim @ GT: mean={cos_proj.mean():.3f}")

# After cross-attention
q_e, r_e = model.cross_attn(q_p, r_p)[:2]
corr = global_correlation(q_e, r_e).reshape(4, 510, 510)

diag = torch.diagonal(corr, dim1=1, dim2=2)  # correct-match scores
mean_all = corr.mean(dim=2)  # average match scores per query

print(f"\nCorrelation volume @ GT pose:")
print(f"  Correct-match (diagonal): {diag.mean():.1f} ± {diag.std():.1f}")
print(f"  All-match (mean per row): {mean_all.mean():.1f} ± {mean_all.std():.1f}")
print(f"  Discrimination gap: {(diag - mean_all).mean():.1f}")

# Top-1 accuracy
top1 = corr.argmax(dim=2)
correct = (top1 == torch.arange(510, device=device)).float().mean()
print(f"  Top-1 accuracy: {correct:.3f}")

# Top-5 accuracy
_, topk = corr.topk(5, dim=2)
correct5 = (topk == torch.arange(510, device=device).unsqueeze(0).unsqueeze(2)).any(dim=2).float().mean()
print(f"  Top-5 accuracy: {correct5:.3f}")

# Also check: flow from v12 model
res = model(q, rf, torch.ones(4, 68, 120, device=device) * 20)
print(f"\nPred flow @ GT pose: mean_mag={res['flow_fine'].norm(dim=1).mean():.4f}px")
print(f"  (Should be ~0 since pose is GT)")

print("\nDone!")
