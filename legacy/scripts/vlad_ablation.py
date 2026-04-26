#!/usr/bin/env python3
"""
AnyLoc 风格 VLAD 全面消融实验
========================================
测试不同参数组合对跨序列检索性能的影响:
  1. norm_descs: True vs False
  2. K 值: 32 vs 64
  3. facet: token vs key vs value (需要重新提取)  
  4. layer: 11 (last) vs 10 (second-to-last)
  5. 当前只能做 K 和 norm 消融 (已有 layer=11 token features)
"""
import numpy as np, torch, faiss, re, time, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from place_recognition.vlad_retrieval import VLADEncoder

def load_patches(dino_dir, traj_path):
    """Load DINO patches and poses."""
    traj = np.loadtxt(traj_path).reshape(-1,4,4).astype(np.float32)
    w2c = np.linalg.inv(traj).astype(np.float32)
    
    file_map = {}
    for fp in Path(dino_dir).glob('rgb_*_fine_dino_*.pt'):
        m = re.search(r'rgb_(\d+)_fine_dino_', fp.name)
        if m:
            fid = int(m.group(1))
            if fid < len(w2c):
                file_map[fid] = fp
    
    fids = sorted(file_map.keys())
    per_frame = []
    for fid in fids:
        f = torch.load(str(file_map[fid]), map_location='cpu', weights_only=True).numpy()
        D, H, W = f.shape
        per_frame.append(f.reshape(D,-1).T.astype(np.float32))
    
    poses = np.stack([w2c[fid] for fid in fids])
    return fids, per_frame, poses


def eval_vlad(enc, db_descs, db_poses, query_data):
    """Evaluate VLAD retrieval."""
    faiss.normalize_L2(db_descs)
    index = faiss.IndexFlatIP(db_descs.shape[1])
    index.add(db_descs)
    
    t_errs = []; r_errs = []
    for fid, patches, gt in query_data:
        v = enc.encode_single(patches).reshape(1,-1)
        faiss.normalize_L2(v)
        _, idx = index.search(v, 1)
        rw = db_poses[int(idx[0,0])]
        # Camera center distance
        C_gt = -gt[:3,:3].T @ gt[:3,3]
        C_ret = -rw[:3,:3].T @ rw[:3,3]
        t_errs.append(float(np.linalg.norm(C_gt - C_ret)))
        R_rel = gt[:3,:3] @ rw[:3,:3].T
        r_errs.append(float(np.degrees(np.arccos(np.clip((np.trace(R_rel)-1)/2, -1, 1)))))
    
    t_errs = np.array(t_errs); r_errs = np.array(r_errs)
    return t_errs, r_errs


def print_results(name, t_errs, r_errs):
    print(f'\n--- {name} ---')
    print(f'  dt: mean={t_errs.mean():.4f}m  med={np.median(t_errs):.4f}m')
    print(f'  dr: mean={r_errs.mean():.2f}°  med={np.median(r_errs):.2f}°')
    for th_t, th_r in [(0.25, 10), (0.5, 15), (0.5, 30), (1.0, 45), (2.0, 60)]:
        r = np.mean((t_errs<th_t)&(r_errs<th_r))*100
        print(f'  <{th_t}m & {th_r}°: {r:.1f}%')
    return {
        'dt_mean': float(t_errs.mean()), 'dt_med': float(np.median(t_errs)),
        'dr_mean': float(r_errs.mean()), 'dr_med': float(np.median(r_errs)),
    }


# === Load Data ===
print("Loading data...")
db_fids, db_frames, db_poses = load_patches(
    'output/features_multiscale/room_0/fine_dino',
    'dataset/room_0/Sequence_1/traj_w_c.txt'
)
print(f"DB: {len(db_fids)} frames, {sum(p.shape[0] for p in db_frames)} total patches")

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

all_patches = np.concatenate(db_frames, axis=0)
print(f"Total patches for K-means: {all_patches.shape}")

# === Ablation experiments ===
results = {}

configs = [
    # (name, K, norm_descs, max_kmeans_samples)
    ("VLAD K=32 no-norm", 32, False, 100000),
    ("VLAD K=32 normed", 32, True, 100000),
    ("VLAD K=64 normed", 64, True, 100000),
    ("VLAD K=16 normed", 16, True, 100000),
    ("VLAD K=32 normed fullsample", 32, True, 500000),
    ("VLAD K=64 normed fullsample", 64, True, 500000),
]

for name, K, norm, max_samples in configs:
    print(f'\n{"="*60}')
    print(f'Experiment: {name}')
    print(f'  K={K}, norm_descs={norm}, max_samples={max_samples}')
    
    enc = VLADEncoder(n_clusters=K, token_dim=768, norm_descs=norm)
    enc.fit(all_patches, max_samples=max_samples, verbose=False)
    
    t0 = time.time()
    db_descs = enc.encode_batch(db_frames)
    elapsed = time.time() - t0
    print(f'  Encoded in {elapsed:.1f}s, desc dim={db_descs.shape[1]}')
    
    t_errs, r_errs = eval_vlad(enc, db_descs, db_poses, qdata)
    r = print_results(name, t_errs, r_errs)
    r['K'] = K
    r['norm'] = norm
    r['max_samples'] = max_samples
    results[name] = r

# === Also test: CLS token baseline for comparison ===
print(f'\n{"="*60}')
print("CLS Token Baseline (from existing features)")
cls_dir = Path('output/features_multiscale/room_0_seq2/cls')
cls_db_dir = Path('output/features_multiscale_compressed/room_0/cls')

# Check if CLS features exist for both
if cls_db_dir.exists() and cls_dir.exists():
    # Load DB CLS
    db_cls = []
    db_cls_fids = []
    for fp in sorted(cls_db_dir.glob('rgb_*_cls_*.pt')):
        m = re.search(r'rgb_(\d+)_cls_', fp.name)
        if m:
            fid = int(m.group(1))
            c = torch.load(str(fp), map_location='cpu', weights_only=True).numpy().flatten()
            # L2 normalize
            c = c / (np.linalg.norm(c) + 1e-8)
            db_cls.append(c)
            db_cls_fids.append(fid)
    
    if db_cls:
        db_cls = np.stack(db_cls).astype(np.float32)
        db_cls_w2c = np.linalg.inv(
            np.loadtxt('dataset/room_0/Sequence_1/traj_w_c.txt').reshape(-1,4,4)
        ).astype(np.float32)
        db_cls_poses = np.stack([db_cls_w2c[fid] for fid in db_cls_fids])
        
        faiss.normalize_L2(db_cls)
        cls_index = faiss.IndexFlatIP(db_cls.shape[1])
        cls_index.add(db_cls)
        
        # Query
        t_errs_cls = []; r_errs_cls = []
        for fp in sorted(cls_dir.glob('rgb_*_cls_*.pt')):
            m_ = re.search(r'rgb_(\d+)_cls_', fp.name)
            if m_:
                qfid = int(m_.group(1))
                if qfid < len(qw2c):
                    q = torch.load(str(fp), map_location='cpu', weights_only=True).numpy().flatten().astype(np.float32)
                    q = q / (np.linalg.norm(q) + 1e-8)
                    q = q.reshape(1,-1)
                    faiss.normalize_L2(q)
                    _, idx = cls_index.search(q, 1)
                    rw = db_cls_poses[int(idx[0,0])]
                    gt = qw2c[qfid]
                    C_gt = -gt[:3,:3].T @ gt[:3,3]
                    C_ret = -rw[:3,:3].T @ rw[:3,3]
                    t_errs_cls.append(float(np.linalg.norm(C_gt - C_ret)))
                    R_rel = gt[:3,:3] @ rw[:3,:3].T
                    r_errs_cls.append(float(np.degrees(np.arccos(np.clip((np.trace(R_rel)-1)/2, -1, 1)))))
        
        t_errs_cls = np.array(t_errs_cls); r_errs_cls = np.array(r_errs_cls)
        print_results("CLS Token (768d)", t_errs_cls, r_errs_cls)
    else:
        print("  No DB CLS features found")
else:
    print(f"  CLS dirs missing: db={cls_db_dir.exists()}, q={cls_dir.exists()}")

# === Summary ===
print(f'\n{"="*60}')
print("SUMMARY")
print(f'{"Method":<35} {"dt_mean":>8} {"dt_med":>8} {"dr_mean":>8} {"dr_med":>8}')
print(f'{"-"*35} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
for name, r in results.items():
    print(f'{name:<35} {r["dt_mean"]:>8.4f} {r["dt_med"]:>8.4f} {r["dr_mean"]:>8.2f} {r["dr_med"]:>8.2f}')
