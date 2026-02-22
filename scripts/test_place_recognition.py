#!/usr/bin/env python3
"""
Phase 3 端到端验证: FAISS 检索质量评估
========================================
1. 构建索引
2. 用自身帧做查询 (self-retrieval), Top-1 应该是自己
3. 评估 Top-K 检索的位姿距离 (平移+旋转)
4. 模拟扰动查询: 添加噪声后观察检索鲁棒性
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from splatloc_modules.place_recognition import PlaceRecognition


def rotation_error_deg(R1, R2):
    """计算两个旋转矩阵的角度误差 (度)."""
    R_rel = R1[:3, :3] @ R2[:3, :3].T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle)


def translation_error(T1, T2):
    """W2C 位姿的平移误差 (m)."""
    return np.linalg.norm(T1[:3, 3] - T2[:3, 3])


def main():
    print("=" * 60)
    print("Phase 3: DINO CLS Token FAISS 检索验证")
    print("=" * 60)

    CLS_DIR = 'output/features_multiscale_compressed/room_0/cls'
    TRAJ_PATH = 'dataset/room_0/Sequence_1/traj_w_c.txt'
    SAVE_PATH = 'output/retrieval/room_0/index.faiss'

    # ── 1. 构建索引 ──
    print("\n[Step 1] 构建 FAISS 索引...")
    retriever = PlaceRecognition.build_from_dir(
        cls_dir=CLS_DIR,
        traj_path=TRAJ_PATH,
    )
    retriever.save(SAVE_PATH)

    # ── 2. 加载验证 ──
    print("\n[Step 2] 从磁盘重新加载索引...")
    retriever2 = PlaceRecognition.load(SAVE_PATH)
    assert len(retriever2) == len(retriever), "加载后帧数不一致"
    print(f"  OK: {retriever2}")

    # ── 3. Self-retrieval 测试 ──
    print(f"\n[Step 3] Self-Retrieval 测试 (用自身 CLS token 查询)...")
    n = len(retriever)
    top1_correct = 0
    top5_contains = 0
    trans_errors_top1 = []
    rot_errors_top1 = []

    # 批量查询所有帧
    all_tokens = retriever.cls_tokens  # [N, 768]
    batch_results = retriever.query_batch(all_tokens, top_k=5)

    for i, results in enumerate(batch_results):
        query_fid = retriever.frame_ids[i]
        query_pose = retriever.poses_w2c[i]

        # Top-1 检查
        top1 = results[0]
        if top1['frame_id'] == query_fid:
            top1_correct += 1

        # Top-5 包含自己
        top5_fids = [r['frame_id'] for r in results]
        if query_fid in top5_fids:
            top5_contains += 1

        # Top-1 位姿误差
        t_err = translation_error(query_pose, top1['pose_w2c'])
        r_err = rotation_error_deg(query_pose, top1['pose_w2c'])
        trans_errors_top1.append(t_err)
        rot_errors_top1.append(r_err)

    print(f"  Top-1 自检索正确率: {top1_correct}/{n} = {top1_correct/n*100:.1f}%")
    print(f"  Top-5 包含自身率:   {top5_contains}/{n} = {top5_contains/n*100:.1f}%")
    print(f"  Top-1 平移误差:     {np.mean(trans_errors_top1):.6f} m (std={np.std(trans_errors_top1):.6f})")
    print(f"  Top-1 旋转误差:     {np.mean(rot_errors_top1):.4f}° (std={np.std(rot_errors_top1):.4f})")

    # ── 4. 跨帧检索: 查询帧 i，看 Top-5 相邻帧 ──
    print(f"\n[Step 4] 跨帧相邻性分析 (Top-5 候选与查询帧的距离)...")
    test_indices = [0, 100, 250, 450, 700, 899]
    for test_idx in test_indices:
        if test_idx >= n:
            continue
        fid = retriever.frame_ids[test_idx]
        token = retriever.cls_tokens[test_idx]
        results = retriever.query(token, top_k=5)

        neighbors = []
        for r in results:
            t_err = translation_error(retriever.poses_w2c[test_idx], r['pose_w2c'])
            neighbors.append(f"#{r['frame_id']}(sim={r['score']:.3f},Δt={t_err:.4f}m)")

        print(f"  Frame {fid}: {', '.join(neighbors)}")

    # ── 5. 噪声鲁棒性测试 ──
    print(f"\n[Step 5] 噪声鲁棒性测试 (CLS token + Gaussian noise)...")
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    for noise_std in noise_levels:
        correct = 0
        for i in range(n):
            token = retriever.cls_tokens[i].copy()
            if noise_std > 0:
                token += np.random.randn(768).astype(np.float32) * noise_std
            results = retriever.query(token, top_k=1)
            if results[0]['frame_id'] == retriever.frame_ids[i]:
                correct += 1
        print(f"  σ={noise_std:.2f}: Top-1 正确率 {correct}/{n} = {correct/n*100:.1f}%")

    # ── 6. 时间基准 ──
    print(f"\n[Step 6] 查询速度基准...")
    import time
    q = retriever.cls_tokens[:1].copy()
    n_queries = 1000
    t0 = time.time()
    for _ in range(n_queries):
        retriever.query(q[0], top_k=5)
    elapsed = time.time() - t0
    print(f"  {n_queries} 次查询耗时: {elapsed*1000:.1f} ms")
    print(f"  单次查询延迟: {elapsed/n_queries*1000:.3f} ms")

    print(f"\n{'='*60}")
    print("Phase 3 验证完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
