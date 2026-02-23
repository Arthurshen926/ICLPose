#!/usr/bin/env python3
"""
NetVLAD Place Recognition (独立实现，不依赖 hloc)
====================================================
使用 VGG16-NetVLAD-Pitts30K 权重做全局描述子提取和检索。

权重来自 hloc 项目:
  https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat

用法:
    # 提取特征 + 构建数据库 + 评估
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python place_recognition/netvlad_retrieval.py \
        --db_image_dir dataset/room_0/Sequence_1/rgb \
        --db_traj_path dataset/room_0/Sequence_1/traj_w_c.txt \
        --query_image_dir dataset/room_0/Sequence_2/rgb \
        --query_traj_path dataset/room_0/Sequence_2/traj_w_c.txt \
        --output_dir output/retrieval/room_0_netvlad \
        --device cuda
"""
import argparse
import logging
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── NetVLAD Layer ────────────────────────────────────────────
class NetVLADLayer(nn.Module):
    """VLAD pooling layer (hard coded K=64, input_dim=512 from VGG16)."""

    def __init__(self, input_dim=512, K=64, score_bias=False, intranorm=True):
        super().__init__()
        self.score_proj = nn.Conv1d(input_dim, K, kernel_size=1, bias=score_bias)
        centers = nn.parameter.Parameter(torch.empty([input_dim, K]))
        nn.init.xavier_uniform_(centers)
        self.register_parameter("centers", centers)
        self.intranorm = intranorm
        self.output_dim = input_dim * K  # 512 * 64 = 32768

    def forward(self, x):
        b = x.size(0)
        scores = self.score_proj(x)
        scores = F.softmax(scores, dim=1)
        diff = x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1)
        desc = (scores.unsqueeze(1) * diff).sum(dim=-1)
        if self.intranorm:
            desc = F.normalize(desc, dim=1)
        desc = desc.view(b, -1)
        desc = F.normalize(desc, dim=1)
        return desc


# ── NetVLAD 模型 ────────────────────────────────────────────
class NetVLADModel(nn.Module):
    """VGG16-NetVLAD with Pitts30K weights → 4096d global descriptor."""

    def __init__(self, weights_path: str = None, whiten: bool = True, device: str = 'cuda'):
        super().__init__()

        # Find weights
        if weights_path is None:
            weights_path = Path(torch.hub.get_dir()) / 'netvlad' / 'VGG16-NetVLAD-Pitts30K.mat'
            if not weights_path.exists():
                print(f"Downloading NetVLAD weights...")
                weights_path.parent.mkdir(exist_ok=True, parents=True)
                url = "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat"
                torch.hub.download_url_to_file(url, str(weights_path))

        # VGG16 backbone (remove last ReLU + MaxPool2d)
        backbone = list(models.vgg16().children())[0]
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # NetVLAD layer
        self.netvlad = NetVLADLayer()

        # Whitening layer → 4096d
        self.do_whiten = whiten
        if whiten:
            self.whiten = nn.Linear(self.netvlad.output_dim, 4096)

        # Load MATLAB weights
        mat = loadmat(str(weights_path), struct_as_record=False, squeeze_me=True)

        # CNN weights
        for layer, mat_layer in zip(self.backbone.children(), mat["net"].layers):
            if isinstance(layer, nn.Conv2d):
                w = torch.tensor(mat_layer.weights[0]).float().permute([3, 2, 0, 1])
                b = torch.tensor(mat_layer.weights[1]).float()
                layer.weight = nn.Parameter(w)
                layer.bias = nn.Parameter(b)

        # NetVLAD weights
        score_w = mat["net"].layers[30].weights[0]  # D x K
        center_w = -mat["net"].layers[30].weights[1]  # D x K (negated in MATLAB)
        self.netvlad.score_proj.weight = nn.Parameter(
            torch.tensor(score_w).float().permute([1, 0]).unsqueeze(-1)
        )
        self.netvlad.centers = nn.Parameter(torch.tensor(center_w).float())

        # Whitening weights
        if whiten:
            w = torch.tensor(mat["net"].layers[33].weights[0]).float().squeeze().permute([1, 0])
            b = torch.tensor(mat["net"].layers[33].weights[1].squeeze()).float()
            self.whiten.weight = nn.Parameter(w)
            self.whiten.bias = nn.Parameter(b)

        # Preprocessing mean (from MATLAB)
        self.register_buffer(
            'pixel_mean',
            torch.tensor(mat["net"].meta.normalization.averageImage[0, 0]).float().view(1, 3, 1, 1)
        )

        self.eval()
        self.to(device)
        self.device = device

        desc_dim = 4096 if whiten else self.netvlad.output_dim
        print(f"[NetVLAD] Loaded VGG16-NetVLAD-Pitts30K")
        print(f"  Weights: {weights_path}")
        print(f"  Descriptor dim: {desc_dim}")
        print(f"  Whiten: {whiten}")

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W] in [0, 1] range
        Returns:
            descriptor: [B, 4096] L2-normalized global descriptor
        """
        # Preprocess: scale to [0, 255] and subtract mean
        x = torch.clamp(image * 255, 0.0, 255.0)
        x = x - self.pixel_mean

        # VGG16 backbone
        feat = self.backbone(x)
        b, c, _, _ = feat.size()
        feat = feat.view(b, c, -1)

        # Pre-normalize + NetVLAD
        feat = F.normalize(feat, dim=1)
        desc = self.netvlad(feat)

        # Whiten + normalize
        if self.do_whiten:
            desc = self.whiten(desc)
            desc = F.normalize(desc, dim=1)

        return desc


# ── 特征提取 ────────────────────────────────────────────────
def extract_descriptors(
    model: NetVLADModel,
    image_dir: str,
    resize_max: int = 1024,
    batch_size: int = 4,
) -> tuple:
    """
    提取目录中所有图像的 NetVLAD 描述子。

    Returns:
        frame_ids: list of int
        descriptors: [N, 4096] np.ndarray
    """
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob('rgb_*.png'))
    if not image_files:
        image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

    transform = T.Compose([
        T.ToTensor(),  # → [0, 1]
    ])

    frame_ids = []
    all_descs = []

    for i in tqdm(range(0, len(image_files), batch_size), desc="NetVLAD提取"):
        batch_files = image_files[i:i + batch_size]
        images = []
        for f in batch_files:
            # Parse frame id
            m = re.search(r'rgb_(\d+)', f.stem)
            fid = int(m.group(1)) if m else i
            frame_ids.append(fid)

            img = Image.open(f).convert('RGB')

            # Resize (keep aspect ratio, max side = resize_max)
            w, h = img.size
            if max(w, h) > resize_max:
                scale = resize_max / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            images.append(transform(img))

        # Pad to same size in batch (NetVLAD is fully convolutional)
        max_h = max(t.shape[1] for t in images)
        max_w = max(t.shape[2] for t in images)
        batch = torch.zeros(len(images), 3, max_h, max_w)
        for j, t in enumerate(images):
            batch[j, :, :t.shape[1], :t.shape[2]] = t

        batch = batch.to(model.device)
        descs = model(batch).cpu().numpy()
        all_descs.append(descs)

    all_descs = np.concatenate(all_descs, axis=0).astype(np.float32)
    return frame_ids, all_descs


# ── 评估 ────────────────────────────────────────────────────
def evaluate_retrieval(
    db_descs: np.ndarray,
    db_poses_w2c: np.ndarray,
    query_descs: np.ndarray,
    query_poses_w2c: np.ndarray,
    top_k: int = 10,
) -> dict:
    """评估检索精度：Camera Center 距离 + 旋转角度。"""

    # Build FAISS index
    d = db_descs.shape[1]
    db_descs_norm = db_descs.copy()
    faiss.normalize_L2(db_descs_norm)
    index = faiss.IndexFlatIP(d)
    index.add(db_descs_norm)

    query_norm = query_descs.copy()
    faiss.normalize_L2(query_norm)
    scores, indices = index.search(query_norm, top_k)

    # Compute errors
    t_errs = []
    r_errs = []
    t_errs_topk = {k: [] for k in [1, 3, 5, 10]}

    for qi in range(len(query_descs)):
        gt = query_poses_w2c[qi]
        C_gt = -gt[:3, :3].T @ gt[:3, 3]

        # Top-1
        ret = db_poses_w2c[int(indices[qi, 0])]
        C_ret = -ret[:3, :3].T @ ret[:3, 3]
        te = float(np.linalg.norm(C_gt - C_ret))
        R_rel = gt[:3, :3] @ ret[:3, :3].T
        re_deg = float(np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))))
        t_errs.append(te)
        r_errs.append(re_deg)

        # Spatial oracle for top-K
        for k in t_errs_topk.keys():
            if k > top_k:
                continue
            topk_idx = indices[qi, :k]
            topk_poses = db_poses_w2c[topk_idx]
            topk_centers = np.stack([-p[:3, :3].T @ p[:3, 3] for p in topk_poses])
            dists = np.linalg.norm(topk_centers - C_gt, axis=1)
            best = np.argmin(dists)
            best_pose = topk_poses[best]
            C_best = -best_pose[:3, :3].T @ best_pose[:3, 3]
            te_k = float(np.linalg.norm(C_gt - C_best))
            R_rel_k = gt[:3, :3] @ best_pose[:3, :3].T
            re_k = float(np.degrees(np.arccos(np.clip((np.trace(R_rel_k) - 1) / 2, -1, 1))))
            t_errs_topk[k].append((te_k, re_k))

    t_errs = np.array(t_errs)
    r_errs = np.array(r_errs)

    results = {
        'top1': {
            'dt_mean': float(t_errs.mean()),
            'dt_med': float(np.median(t_errs)),
            'dr_mean': float(r_errs.mean()),
            'dr_med': float(np.median(r_errs)),
        },
        'thresholds': {},
    }

    print(f"\n{'='*60}")
    print(f"NetVLAD Retrieval Results ({len(query_descs)} queries → {len(db_descs)} DB)")
    print(f"{'='*60}")
    print(f"Top-1: dt_mean={t_errs.mean():.4f}m  dt_med={np.median(t_errs):.4f}m  "
          f"dr_mean={r_errs.mean():.2f}°  dr_med={np.median(r_errs):.2f}°")

    for th_t, th_r in [(0.25, 10), (0.25, 15), (0.5, 30), (1.0, 45), (2.0, 60)]:
        pct = np.mean((t_errs < th_t) & (r_errs < th_r)) * 100
        print(f"  <{th_t}m & {th_r}°: {pct:.1f}%")
        results['thresholds'][f'{th_t}m_{th_r}deg'] = pct

    # Spatial oracle top-K
    print(f"\nSpatial Oracle from Top-K:")
    for k in sorted(t_errs_topk.keys()):
        if k > top_k or not t_errs_topk[k]:
            continue
        tk_arr = np.array(t_errs_topk[k])
        t_k, r_k = tk_arr[:, 0], tk_arr[:, 1]
        p1 = np.mean((t_k < 0.5) & (r_k < 30)) * 100
        p2 = np.mean((t_k < 1.0) & (r_k < 45)) * 100
        print(f"  Top-{k:>2}: dt={t_k.mean():.3f}m  dr={r_k.mean():.1f}°  "
              f"<0.5m&30°={p1:.1f}%  <1.0m&45°={p2:.1f}%")
        results[f'top{k}_oracle'] = {
            'dt_mean': float(t_k.mean()), 'dr_mean': float(r_k.mean()),
        }

    return results


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='NetVLAD Place Recognition')
    parser.add_argument('--db_image_dir', type=str, required=True)
    parser.add_argument('--db_traj_path', type=str, required=True)
    parser.add_argument('--query_image_dir', type=str, required=True)
    parser.add_argument('--query_traj_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output/retrieval/room_0_netvlad')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to Pitts30K_struct.mat (auto-download if not set)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resize_max', type=int, default=640,
                        help='Max side for input images (NetVLAD default 1024, we use 640 for speed)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = NetVLADModel(weights_path=args.weights_path, device=args.device)

    # Extract DB descriptors (with caching)
    db_cache = out_dir / 'db_descriptors.npy'
    fid_cache = out_dir / 'db_frame_ids.npy'
    if db_cache.exists() and fid_cache.exists():
        print(f"\n--- Database: loading cached descriptors ---")
        db_descs = np.load(db_cache)
        db_fids = np.load(fid_cache).tolist()
        print(f"  {len(db_fids)} frames, {db_descs.shape[1]}d (cached)")
    else:
        print(f"\n--- Database: {args.db_image_dir} ---")
        t0 = time.time()
        db_fids, db_descs = extract_descriptors(model, args.db_image_dir,
                                                resize_max=args.resize_max,
                                                batch_size=args.batch_size)
        print(f"  {len(db_fids)} frames, {db_descs.shape[1]}d, {time.time()-t0:.1f}s")
        np.save(db_cache, db_descs)
        np.save(fid_cache, np.array(db_fids))

    # Extract Query descriptors (with caching)
    q_cache = out_dir / 'query_descriptors.npy'
    qfid_cache = out_dir / 'query_frame_ids.npy'
    if q_cache.exists() and qfid_cache.exists():
        print(f"\n--- Query: loading cached descriptors ---")
        q_descs = np.load(q_cache)
        q_fids = np.load(qfid_cache).tolist()
        print(f"  {len(q_fids)} frames, {q_descs.shape[1]}d (cached)")
    else:
        print(f"\n--- Query: {args.query_image_dir} ---")
        t0 = time.time()
        q_fids, q_descs = extract_descriptors(model, args.query_image_dir,
                                              resize_max=args.resize_max,
                                              batch_size=args.batch_size)
        print(f"  {len(q_fids)} frames, {q_descs.shape[1]}d, {time.time()-t0:.1f}s")
        np.save(q_cache, q_descs)
        np.save(qfid_cache, np.array(q_fids))

    # Load poses
    db_traj = np.loadtxt(args.db_traj_path).reshape(-1, 4, 4).astype(np.float32)
    db_w2c = np.linalg.inv(db_traj).astype(np.float32)
    db_poses = np.stack([db_w2c[fid] for fid in db_fids if fid < len(db_w2c)])

    q_traj = np.loadtxt(args.query_traj_path).reshape(-1, 4, 4).astype(np.float32)
    q_w2c = np.linalg.inv(q_traj).astype(np.float32)
    q_poses = np.stack([q_w2c[fid] for fid in q_fids if fid < len(q_w2c)])

    # Evaluate
    import json
    results = evaluate_retrieval(db_descs, db_poses, q_descs, q_poses, top_k=args.top_k)

    # Save
    np.save(out_dir / 'db_descriptors.npy', db_descs)
    np.save(out_dir / 'query_descriptors.npy', q_descs)
    np.save(out_dir / 'db_poses_w2c.npy', db_poses)
    np.save(out_dir / 'query_poses_w2c.npy', q_poses)
    np.save(out_dir / 'db_frame_ids.npy', np.array(db_fids))
    np.save(out_dir / 'query_frame_ids.npy', np.array(q_fids))
    print(f"\nSaved to {out_dir}")


if __name__ == '__main__':
    main()
