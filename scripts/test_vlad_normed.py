#!/usr/bin/env python3
"""Test VLAD with norm_descs=True (AnyLoc-style fix)."""
import numpy as np, torch, faiss, re, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from place_recognition.vlad_retrieval import VLADEncoder

# === Build VLAD from Seq1 real features ===
dino_dir = Path('output/features_multiscale/room_0/fine_dino')
traj = np.loadtxt('dataset/room_0/Sequence_1/traj_w_c.txt').reshape(-1,4,4).astype(np.float32)
w2c = np.linalg.inv(traj).astype(np.float32)

file_map = {}
for fp in dino_dir.glob('rgb_*_fine_dino_*.pt'):
    m = re.search(r'rgb_(\d+)_fine_dino_', fp.name)
    if m:
        fid = int(m.group(1))
        if fid < len(w2c):
            file_map[fid] = fp

fids = sorted(file_map.keys())
print(f'DB frames: {len(fids)}')

all_patches = []
per_frame = []
for fid in fids:
    f = torch.load(str(file_map[fid]), map_location='cpu', weights_only=True).numpy()
    D, H, W = f.shape
    p = f.reshape(D,-1).T.astype(np.float32)
    all_patches.append(p)
    per_frame.append(p)

all_patches_cat = np.concatenate(all_patches, axis=0)
print(f'Total patches: {all_patches_cat.shape}')

# Build encoder with norm_descs=True (AnyLoc style)
enc = VLADEncoder(n_clusters=32, token_dim=768, norm_descs=True)
enc.fit(all_patches_cat, max_samples=100000, verbose=True)

# Encode all DB frames
t0 = time.time()
db_descs = enc.encode_batch(per_frame)
print(f'Encoded {len(fids)} frames in {time.time()-t0:.1f}s, shape={db_descs.shape}')

# Build FAISS
faiss.normalize_L2(db_descs)
index = faiss.IndexFlatIP(db_descs.shape[1])
index.add(db_descs)
db_poses = np.stack([w2c[fid] for fid in fids])

# === Query with Seq2 ===
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

print(f'\nQueries: {len(qdata)}')
t_errs = []; r_errs = []
for fid, patches, gt in qdata:
    v = enc.encode_single(patches).reshape(1,-1)
    faiss.normalize_L2(v)
    sc, idx = index.search(v, 1)
    rw = db_poses[int(idx[0,0])]
    C_gt = -gt[:3,:3].T @ gt[:3,3]
    C_ret = -rw[:3,:3].T @ rw[:3,3]
    t_errs.append(float(np.linalg.norm(C_gt - C_ret)))
    R_rel = gt[:3,:3] @ rw[:3,:3].T
    r_errs.append(float(np.degrees(np.arccos(np.clip((np.trace(R_rel)-1)/2, -1, 1)))))

t_errs = np.array(t_errs); r_errs = np.array(r_errs)
print(f'\n=== VLAD (norm_descs=True, AnyLoc-style) ===')
print(f'dt: mean={t_errs.mean():.4f}m  med={np.median(t_errs):.4f}m')
print(f'dr: mean={r_errs.mean():.2f}deg  med={np.median(r_errs):.2f}deg')
for th_t, th_r in [(0.25, 15), (0.5, 30), (1.0, 45), (2.0, 60)]:
    print(f'  <{th_t}m & {th_r}deg: {np.mean((t_errs<th_t)&(r_errs<th_r))*100:.1f}%')

# Save
out = Path('output/retrieval/room_0_vlad_normed')
out.mkdir(parents=True, exist_ok=True)
enc.save(str(out / 'vlad_encoder'))
faiss.write_index(index, str(out / 'vlad_index.faiss'))
np.save(out / 'poses_w2c.npy', db_poses)
np.save(out / 'frame_ids.npy', np.array(fids))
np.save(out / 'descriptors.npy', db_descs)
print(f'\nSaved to {out}')
