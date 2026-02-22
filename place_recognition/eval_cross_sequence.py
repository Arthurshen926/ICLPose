#!/usr/bin/env python3
"""
跨序列检索验证: 用 Sequence_2 查询 Sequence_1 的 FAISS 索引
===========================================================
验证 DINO CLS Token 的真实 place recognition 能力:
  - Seq1 建库 (900帧), Seq2 做查询 (900帧)
  - 评估 Top-K 检索的位姿误差 (Seq2 GT 位姿 vs Seq1 检索位姿)
"""
import sys
import re
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from place_recognition.cls_retrieval import PlaceRecognition


def rotation_error_deg(R1, R2):
    """两个 3x3 旋转矩阵之间的角度误差 (度)"""
    R_rel = R1[:3, :3] @ R2[:3, :3].T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle)


def translation_error(T1, T2):
    """两个 4x4 位姿之间的平移误差 (m)"""
    return np.linalg.norm(T1[:3, 3] - T2[:3, 3])


def main():
    print("=" * 65)
    print("跨序列检索验证: Seq2 → Seq1")
    print("=" * 65)

    # ── 路径配置 ──
    SEQ1_CLS_DIR = 'output/features_multiscale_compressed/room_0/cls'
    SEQ1_TRAJ = 'dataset/room_0/Sequence_1/traj_w_c.txt'
    SEQ2_CLS_DIR = 'output/features_multiscale_compressed/room_0_seq2/cls'
    SEQ2_TRAJ = 'dataset/room_0/Sequence_2/traj_w_c.txt'
    INDEX_PATH = 'output/retrieval/room_0/index.faiss'

    # ── 1. 加载/构建 Seq1 索引 ──
    print("\n[Step 1] 加载 Seq1 索引...")
    if Path(INDEX_PATH).exists():
        retriever = PlaceRecognition.load(INDEX_PATH)
    else:
        retriever = PlaceRecognition.build_from_dir(
            cls_dir=SEQ1_CLS_DIR, traj_path=SEQ1_TRAJ
        )
        retriever.save(INDEX_PATH)

    # ── 2. 加载 Seq2 的 CLS tokens + GT 位姿 ──
    print("\n[Step 2] 加载 Seq2 数据...")
    traj2 = np.loadtxt(SEQ2_TRAJ)
    c2w_seq2 = traj2.reshape(-1, 4, 4).astype(np.float32)
    w2c_seq2 = np.linalg.inv(c2w_seq2).astype(np.float32)

    cls_dir = Path(SEQ2_CLS_DIR)
    seq2_data = []
    for fpath in sorted(cls_dir.glob('rgb_*_cls_*.pt')):
        match = re.search(r'rgb_(\d+)_cls_', fpath.name)
        if match:
            fid = int(match.group(1))
            if fid < len(w2c_seq2):
                token = torch.load(fpath, map_location='cpu').numpy().astype(np.float32)
                seq2_data.append((fid, token, w2c_seq2[fid]))

    print(f"  Seq2 帧数: {len(seq2_data)}")

    # ── 3. 批量查询 ──
    print("\n[Step 3] 跨序列批量检索 (Top-K=1,5,10)...")
    all_tokens = np.stack([d[1] for d in seq2_data])

    for top_k in [1, 5, 10]:
        batch_results = retriever.query_batch(all_tokens, top_k=top_k)

        trans_errors = []
        rot_errors = []

        for i, (fid2, token2, gt_pose2) in enumerate(seq2_data):
            # 取 Top-1 的位姿
            best = batch_results[i][0]
            t_err = translation_error(gt_pose2, best['pose_w2c'])
            r_err = rotation_error_deg(gt_pose2, best['pose_w2c'])
            trans_errors.append(t_err)
            rot_errors.append(r_err)

        trans_arr = np.array(trans_errors)
        rot_arr = np.array(rot_errors)

        print(f"\n  === Top-{top_k} 检索 (取最佳候选的位姿) ===")
        print(f"  平移误差: mean={trans_arr.mean():.4f}m  median={np.median(trans_arr):.4f}m  "
              f"std={trans_arr.std():.4f}  max={trans_arr.max():.4f}m")
        print(f"  旋转误差: mean={rot_arr.mean():.2f}°  median={np.median(rot_arr):.2f}°  "
              f"std={rot_arr.std():.2f}  max={rot_arr.max():.2f}°")

        # 精度阈值统计
        for t_th, r_th in [(0.05, 5), (0.1, 10), (0.25, 15), (0.5, 30)]:
            pct = np.mean((trans_arr < t_th) & (rot_arr < r_th)) * 100
            print(f"    < {t_th}m & {r_th}°: {pct:.1f}%")

    # ── 4. 最相似度分布 ──
    print("\n[Step 4] 相似度分布统计...")
    batch_results_5 = retriever.query_batch(all_tokens, top_k=5)
    top1_scores = [r[0]['score'] for r in batch_results_5]
    top5_scores = [r[-1]['score'] for r in batch_results_5]
    top1_arr = np.array(top1_scores)
    top5_arr = np.array(top5_scores)
    print(f"  Top-1 相似度: mean={top1_arr.mean():.4f}  min={top1_arr.min():.4f}  max={top1_arr.max():.4f}")
    print(f"  Top-5 相似度: mean={top5_arr.mean():.4f}  min={top5_arr.min():.4f}  max={top5_arr.max():.4f}")

    # ── 5. 展示一些样例 ──
    print("\n[Step 5] 样例检索结果 (每隔 100 帧)...")
    for i in range(0, len(seq2_data), 100):
        fid2, _, gt_pose2 = seq2_data[i]
        results = batch_results_5[i]
        top1 = results[0]
        t_err = translation_error(gt_pose2, top1['pose_w2c'])
        r_err = rotation_error_deg(gt_pose2, top1['pose_w2c'])
        neighbors = [f"Seq1#{r['frame_id']}(sim={r['score']:.3f})" for r in results[:3]]
        print(f"  Seq2#{fid2:3d} → {', '.join(neighbors)}  | Δt={t_err:.4f}m  Δr={r_err:.2f}°")

    print(f"\n{'='*65}")
    print("跨序列验证完成!")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
