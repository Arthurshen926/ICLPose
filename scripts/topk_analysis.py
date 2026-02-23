#!/usr/bin/env python3
"""Top-K Recall + Spatial Oracle Analysis."""
import numpy as np, torch, faiss, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from place_recognition.vlad_retrieval import VLADEncoder

# Load DB
dino_dir = Path('output/features_multiscale/room_0/fine_dino')
traj = np.loadtxt('dataset/room_0/Sequence_1/traj_w_c.txt').reshape(-1,4,4).astype(np.float32)
w2c_db = np.linalg.inv(traj).astype(np.float32)

file_map = {}
for fp in dino_dir.glob('rgb_*_fine_dino_*.pt'):
    m = re.search(r'rgb_(\d+)_fine_dino_', fp.name)
    if m:
        fid = int(m.group(1))
        if fid < len(w2c_db):
            file_map[fid] = fp

fids = sorted(file_map.keys())
per_frame = []
for fid in fids:
    f = torch.load(str(file_map[fid]), map_location='cpu', weights_only=True).numpy()
    per_frame.append(f.reshape(f.shape[0],-1).T.astype(np.float32))

print(f"DB: {len(fids)} frames")

# K=64 normed VLAD
all_patches = np.concatenate(per_frame, axis=0)
enc = VLADEncoder(n_clusters=64, token_dim=768, norm_descs=True)
enc.fit(all_patches, max_samples=100000, verbose=True)
db_descs = enc.encode_batch(per_frame)
faiss.normalize_L2(db_descs)
index = faiss.IndexFlatIP(db_descs.shape[1])
index.add(db_descs)
db_poses = np.stack([w2c_db[fid] for fid in fids])
db_centers = np.stack([-p[:3,:3].T @ p[:3,3] for p in db_poses])

# Query
qt = np.loadtxt('dataset/room_0/Sequence_2/traj_w_c.txt').reshape(-1,4,4).astype(np.float32)
qw2c = np.linalg.inv(qt).astype(np.float32)
qdir = Path('output/features_multiscale/room_0_seq2/fine_dino')
qdata = []
for fp in sorted(qdir.glob('rgb_*_fine_dino_*.pt')):
    m = re.search(r'rgb_(\d+)_fine_dino_', fp.name)
    if m:
        fid = int(m.group(1))
        if fid < len(qw2c):
            f = torch.load(str(fp), map_location='cpu', weights_only=True).numpy()
            qdata.append((fid, f.reshape(f.shape[0],-1).T.astype(np.float32), qw2c[fid]))

print(f"Queries: {len(qdata)}")

# Encode all queries first
TOP_K = 20
q_vlads = []
for fid, patches, gt in qdata:
    v = enc.encode_single(patches).reshape(1,-1)
    faiss.normalize_L2(v)
    q_vlads.append(v)
q_vlads = np.concatenate(q_vlads, axis=0)
all_scores, all_indices = index.search(q_vlads, TOP_K)

print(f"\n{'Top-K':<8} {'Visual Top-1':>35} | {'Spatial Oracle from Top-K':>50}")
print(f"{'':>8} {'dt':>8} {'dr':>8} {'<.5m&30':>8} {'<1m&45':>8} | {'dt':>8} {'dr':>8} {'<.5m&30':>8} {'<1m&45':>8}")
print("-" * 105)

for k_pick in [1, 3, 5, 10, 20]:
    t_v = []; r_v = []; t_s = []; r_s = []
    
    for qi, (fid, patches, gt) in enumerate(qdata):
        C_gt = -gt[:3,:3].T @ gt[:3,3]
        
        # Visual top-1
        rw = db_poses[int(all_indices[qi, 0])]
        C_ret = -rw[:3,:3].T @ rw[:3,3]
        t_v.append(float(np.linalg.norm(C_gt - C_ret)))
        R_rel = gt[:3,:3] @ rw[:3,:3].T
        r_v.append(float(np.degrees(np.arccos(np.clip((np.trace(R_rel)-1)/2, -1, 1)))))
        
        # Spatial oracle from top-K
        topk_idx = all_indices[qi, :k_pick]
        topk_centers = db_centers[topk_idx]
        dists = np.linalg.norm(topk_centers - C_gt, axis=1)
        best = np.argmin(dists)
        rw_s = db_poses[int(topk_idx[best])]
        C_ret_s = -rw_s[:3,:3].T @ rw_s[:3,3]
        t_s.append(float(np.linalg.norm(C_gt - C_ret_s)))
        R_rel_s = gt[:3,:3] @ rw_s[:3,:3].T
        r_s.append(float(np.degrees(np.arccos(np.clip((np.trace(R_rel_s)-1)/2, -1, 1)))))
    
    t_v = np.array(t_v); r_v = np.array(r_v)
    t_s = np.array(t_s); r_s = np.array(r_s)
    
    p1_v = np.mean((t_v < 0.5) & (r_v < 30)) * 100
    p2_v = np.mean((t_v < 1.0) & (r_v < 45)) * 100
    p1_s = np.mean((t_s < 0.5) & (r_s < 30)) * 100
    p2_s = np.mean((t_s < 1.0) & (r_s < 45)) * 100
    
    print(f"Top-{k_pick:<3} {t_v.mean():>8.3f} {r_v.mean():>8.1f} {p1_v:>7.1f}% {p2_v:>7.1f}% | {t_s.mean():>8.3f} {r_s.mean():>8.1f} {p1_s:>7.1f}% {p2_s:>7.1f}%")

# Also show: from top-20, what's the best possible rotation match?
print(f"\n--- Rotation oracle from Top-20 ---")
t_ro = []; r_ro = []
for qi, (fid, patches, gt) in enumerate(qdata):
    C_gt = -gt[:3,:3].T @ gt[:3,3]
    topk_idx = all_indices[qi, :20]
    best_re = 999
    best_bi = 0
    for bi in topk_idx:
        rw = db_poses[int(bi)]
        R_rel = gt[:3,:3] @ rw[:3,:3].T
        re_ = float(np.degrees(np.arccos(np.clip((np.trace(R_rel)-1)/2, -1, 1))))
        if re_ < best_re:
            best_re = re_
            best_bi = int(bi)
    rw = db_poses[best_bi]
    C_ret = -rw[:3,:3].T @ rw[:3,3]
    t_ro.append(float(np.linalg.norm(C_gt - C_ret)))
    r_ro.append(best_re)

t_ro = np.array(t_ro); r_ro = np.array(r_ro)
print(f"  dt mean={t_ro.mean():.3f}m  dr mean={r_ro.mean():.1f}°")
print(f"  <0.5m&30°: {np.mean((t_ro<0.5)&(r_ro<30))*100:.1f}%  <1m&45°: {np.mean((t_ro<1.0)&(r_ro<45))*100:.1f}%")
