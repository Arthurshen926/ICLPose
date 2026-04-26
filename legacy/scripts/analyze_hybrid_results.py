#!/usr/bin/env python3
"""Analyze hybrid retrieval results by source type (real vs rendered)."""
import numpy as np, faiss, torch, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from place_recognition.vlad_retrieval import VLADEncoder

d = Path('output/retrieval/room_0_hybrid')
index = faiss.read_index(str(d / 'augmented_index.faiss'))
poses = np.load(d / 'poses_w2c.npy')
is_orig = np.load(d / 'is_original.npy')
enc = VLADEncoder.load(str(d / 'vlad_encoder'))

qt = np.loadtxt('dataset/room_0/Sequence_2/traj_w_c.txt').reshape(-1,4,4).astype(np.float32)
qw2c = np.linalg.inv(qt).astype(np.float32)

qdir = Path('output/features_multiscale/room_0_seq2/fine_dino')
qdata = []
for fp in sorted(qdir.glob('rgb_*_fine_dino_*.pt')):
    m = re.search(r'rgb_(\d+)_fine_dino_', fp.name)
    if m:
        fid = int(m.group(1))
        if fid < len(qw2c):
            f = torch.load(str(fp), map_location='cpu').numpy()
            qdata.append((fid, f.reshape(f.shape[0],-1).T, qw2c[fid]))

print(f"Loaded {len(qdata)} queries")
print(f"DB: {len(poses)} entries, {is_orig.sum()} real, {(~is_orig).sum()} rendered")

n_real_hit = 0; n_aug_hit = 0
t_err_real = []; t_err_aug = []; r_err_real = []; r_err_aug = []

for fid, patches, gt in qdata:
    v = enc.encode_single(patches.astype(np.float32)).reshape(1,-1)
    faiss.normalize_L2(v)
    sc, idx = index.search(v, 1)
    bi = int(idx[0,0])
    rw = poses[bi]
    # Camera center distance
    C_gt = -gt[:3,:3].T @ gt[:3,3]
    C_ret = -rw[:3,:3].T @ rw[:3,3]
    te = float(np.linalg.norm(C_gt - C_ret))
    R_rel = gt[:3,:3] @ rw[:3,:3].T
    re_deg = float(np.degrees(np.arccos(np.clip((np.trace(R_rel)-1)/2, -1, 1))))
    if is_orig[bi]:
        n_real_hit += 1; t_err_real.append(te); r_err_real.append(re_deg)
    else:
        n_aug_hit += 1; t_err_aug.append(te); r_err_aug.append(re_deg)

t_real = np.array(t_err_real) if t_err_real else np.array([0.])
t_aug = np.array(t_err_aug) if t_err_aug else np.array([0.])
r_real = np.array(r_err_real) if r_err_real else np.array([0.])
r_aug = np.array(r_err_aug) if r_err_aug else np.array([0.])
t_all = np.concatenate([t_real, t_aug])
r_all = np.concatenate([r_real, r_aug])

print(f"\nQuery hit分布: real={n_real_hit} ({n_real_hit/len(qdata)*100:.1f}%), aug={n_aug_hit} ({n_aug_hit/len(qdata)*100:.1f}%)")
print(f"\n--- Camera Center 距离 ---")
print(f"  全部: dt_mean={t_all.mean():.4f}m  dt_med={np.median(t_all):.4f}m  dr_mean={r_all.mean():.2f}°  dr_med={np.median(r_all):.2f}°")
if t_err_real:
    print(f"  真实: dt_mean={t_real.mean():.4f}m  dt_med={np.median(t_real):.4f}m  dr_mean={r_real.mean():.2f}°  dr_med={np.median(r_real):.2f}°")
if t_err_aug:
    print(f"  渲染: dt_mean={t_aug.mean():.4f}m  dt_med={np.median(t_aug):.4f}m  dr_mean={r_aug.mean():.2f}°  dr_med={np.median(r_aug):.2f}°")

print(f"\n--- 阈值统计 ---")
for th_t, th_r in [(0.25, 15), (0.5, 30), (1.0, 45), (2.0, 60)]:
    pct = np.mean((t_all < th_t) & (r_all < th_r)) * 100
    pct_r = np.mean((t_real < th_t) & (r_real < th_r)) * 100 if len(t_real) > 0 else 0
    pct_a = np.mean((t_aug < th_t) & (r_aug < th_r)) * 100 if len(t_aug) > 0 else 0
    print(f"  <{th_t}m & {th_r}°: all={pct:.1f}%  real={pct_r:.1f}%  aug={pct_a:.1f}%")

# Also compare: what if we only use real frames (no augmentation)?
print(f"\n--- 对比: 仅用真实帧的 VLAD 检索 ---")
# Filter out augmented entries
real_mask = is_orig
real_descs = np.load(d / 'descriptors.npy')[real_mask]
real_poses = poses[real_mask]
real_index = faiss.IndexFlatIP(real_descs.shape[1])
faiss.normalize_L2(real_descs)
real_index.add(real_descs)

t_errs_ro = []; r_errs_ro = []
for fid, patches, gt in qdata:
    v = enc.encode_single(patches.astype(np.float32)).reshape(1,-1)
    faiss.normalize_L2(v)
    sc, idx = real_index.search(v, 1)
    bi = int(idx[0,0])
    rw = real_poses[bi]
    C_gt = -gt[:3,:3].T @ gt[:3,3]
    C_ret = -rw[:3,:3].T @ rw[:3,3]
    t_errs_ro.append(float(np.linalg.norm(C_gt - C_ret)))
    R_rel = gt[:3,:3] @ rw[:3,:3].T
    r_errs_ro.append(float(np.degrees(np.arccos(np.clip((np.trace(R_rel)-1)/2, -1, 1)))))

t_ro = np.array(t_errs_ro); r_ro = np.array(r_errs_ro)
print(f"  Real-only VLAD: dt_mean={t_ro.mean():.4f}m  dt_med={np.median(t_ro):.4f}m  dr_mean={r_ro.mean():.2f}°  dr_med={np.median(r_ro):.2f}°")
for th_t, th_r in [(0.25, 15), (0.5, 30), (1.0, 45), (2.0, 60)]:
    pct = np.mean((t_ro < th_t) & (r_ro < th_r)) * 100
    print(f"  <{th_t}m & {th_r}°: {pct:.1f}%")
