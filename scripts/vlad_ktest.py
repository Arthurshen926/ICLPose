#!/usr/bin/env python3
"""Quick VLAD K-value comparison"""
import sys, numpy as np, torch, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from splatloc_modules.vlad_place_recognition import VLADPlaceRecognition

seq1_dir = 'output/features_multiscale/room_0'
seq2_dir = 'output/features_multiscale/room_0_seq2'
seq1_traj = 'dataset/room_0/Sequence_1/traj_w_c.txt'
seq2_traj = 'dataset/room_0/Sequence_2/traj_w_c.txt'

traj2 = np.loadtxt(seq2_traj).reshape(-1,4,4).astype(np.float32)
w2c2 = np.linalg.inv(traj2).astype(np.float32)

dino_dir = Path(seq2_dir) / 'fine_dino'
q_feats, q_poses = [], []
for fp in sorted(dino_dir.glob('rgb_*_fine_dino_*.pt')):
    m = re.search(r'rgb_(\d+)_fine_dino_', fp.name)
    if m:
        fid = int(m.group(1))
        if fid < len(w2c2):
            q_feats.append(torch.load(str(fp), map_location='cpu').numpy())
            q_poses.append(w2c2[fid])

def pose_err(p1, p2):
    c1 = -p1[:3,:3].T @ p1[:3,3]
    c2 = -p2[:3,:3].T @ p2[:3,3]
    dt = np.linalg.norm(c1-c2)
    R = p1[:3,:3] @ p2[:3,:3].T
    cos_a = np.clip((np.trace(R)-1)/2, -1, 1)
    return dt, np.degrees(np.arccos(cos_a))

print(f"Loaded {len(q_feats)} query frames")
for K in [16, 32, 48, 64]:
    t0 = time.time()
    s = VLADPlaceRecognition.build_from_features(
        feature_dir=seq1_dir, traj_path=seq1_traj,
        n_clusters=K, verbose=False)
    bt = time.time() - t0
    
    te, re_list, sc = [], [], []
    t0 = time.time()
    for i in range(len(q_feats)):
        r = s.query(q_feats[i], top_k=1)[0]
        dt, dr = pose_err(r['pose_w2c'], q_poses[i])
        te.append(dt); re_list.append(dr); sc.append(r['score'])
    qt = time.time() - t0
    
    suc = sum(1 for t,r in zip(te, re_list) if t<0.5 and r<10)/len(te)*100
    print(f'K={K:3d} | {K*768:5d}d | Δt={np.mean(te):.3f}m Δr={np.mean(re_list):.1f}° '
          f'| Suc={suc:.1f}% | score={np.mean(sc):.4f} | {bt:.0f}s/{qt:.0f}s')
