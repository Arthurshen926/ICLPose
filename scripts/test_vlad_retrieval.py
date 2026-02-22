#!/usr/bin/env python
"""
Build VLAD Index + Cross-Sequence Retrieval Evaluation
=======================================================
1. 从 Seq1 的 DINO patch tokens 构建 VLAD 索引
2. 用 Seq2 查询，评估跨序列检索精度
3. 对比 CLS Token vs VLAD 的检索效果

用法:
    PYTHONPATH=. python scripts/test_vlad_retrieval.py

输出:
    output/retrieval/room_0_vlad/
    ├── vlad_encoder/       # VLAD 聚类中心
    ├── vlad_index.faiss    # FAISS 索引
    ├── eval_results.json   # 跨序列评测结果
    └── comparison.txt      # CLS vs VLAD 对比
"""

import sys
import json
import numpy as np
import torch
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from splatloc_modules.vlad_place_recognition import VLADPlaceRecognition
from splatloc_modules.place_recognition import PlaceRecognition


def compute_pose_error(pose_pred, pose_gt):
    """计算两个 W2C 位姿之间的平移和旋转误差。"""
    # 位姿是 W2C [4,4]
    # 提取相机位置 (world frame)
    R_pred, t_pred = pose_pred[:3, :3], pose_pred[:3, 3]
    R_gt, t_gt = pose_gt[:3, :3], pose_gt[:3, 3]

    # 相机在世界坐标系中的位置: C = -R^T @ t
    cam_pred = -R_pred.T @ t_pred
    cam_gt = -R_gt.T @ t_gt
    trans_err = np.linalg.norm(cam_pred - cam_gt)

    # 旋转误差
    R_rel = R_pred @ R_gt.T
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    rot_err = np.degrees(np.arccos(cos_angle))

    return trans_err, rot_err


def evaluate_retrieval(retriever, query_features, query_poses_w2c,
                       query_frame_ids, db_poses_w2c, db_frame_ids,
                       top_k=5, method_name="", is_vlad=False):
    """
    评估检索精度。

    Args:
        retriever: PlaceRecognition 或 VLADPlaceRecognition
        query_features: 查询特征 (CLS tokens 或 DINO patch features)
        query_poses_w2c: [N_q, 4, 4] 查询位姿
        query_frame_ids: 查询帧 ID 列表
        db_poses_w2c: [N_db, 4, 4] 数据库位姿
        db_frame_ids: 数据库帧 ID 列表
        top_k: 评估 Top-K
        method_name: 方法名称
        is_vlad: 是否为 VLAD 检索
    """
    n_query = len(query_features)
    trans_errors = {k: [] for k in range(1, top_k + 1)}
    rot_errors = {k: [] for k in range(1, top_k + 1)}
    top1_scores = []

    t_start = time.time()

    for i in range(n_query):
        if is_vlad:
            results = retriever.query(query_features[i], top_k=top_k)
        else:
            results = retriever.query(query_features[i], top_k=top_k)

        q_pose = query_poses_w2c[i]

        if results:
            top1_scores.append(results[0]['score'])

        for r in results:
            rank = r['rank'] + 1  # 1-indexed
            if rank <= top_k:
                t_err, r_err = compute_pose_error(r['pose_w2c'], q_pose)
                trans_errors[rank].append(t_err)
                rot_errors[rank].append(r_err)

    elapsed = time.time() - t_start

    # 汇总
    print(f"\n{'='*60}")
    print(f"[{method_name}] 跨序列检索结果 ({n_query} 查询)")
    print(f"{'='*60}")

    results_summary = {'method': method_name, 'n_queries': n_query, 'time_s': elapsed}

    for k in range(1, min(4, top_k + 1)):
        if trans_errors[k]:
            mean_t = np.mean(trans_errors[k])
            mean_r = np.mean(rot_errors[k])
            median_t = np.median(trans_errors[k])
            median_r = np.median(rot_errors[k])
            # 成功率: Δt < 0.5m 且 Δr < 10°
            success = sum(1 for t, r in zip(trans_errors[k], rot_errors[k])
                         if t < 0.5 and r < 10.0)
            success_rate = success / len(trans_errors[k]) * 100

            print(f"  Top-{k}: Δt mean={mean_t:.3f}m (med={median_t:.3f}m) "
                  f"Δr mean={mean_r:.1f}° (med={median_r:.1f}°) "
                  f"Success(<0.5m,<10°): {success_rate:.1f}%")

            results_summary[f'top{k}'] = {
                'mean_t': float(mean_t), 'median_t': float(median_t),
                'mean_r': float(mean_r), 'median_r': float(median_r),
                'success_rate': float(success_rate),
            }

    if top1_scores:
        print(f"  Top-1 相似度: mean={np.mean(top1_scores):.4f} "
              f"min={np.min(top1_scores):.4f} max={np.max(top1_scores):.4f}")
        results_summary['top1_score'] = {
            'mean': float(np.mean(top1_scores)),
            'min': float(np.min(top1_scores)),
            'max': float(np.max(top1_scores)),
        }

    print(f"  检索耗时: {elapsed:.2f}s ({elapsed/n_query*1000:.1f}ms/query)")
    return results_summary


def main():
    # ============================================================
    # 配置
    # ============================================================
    seq1_feature_dir = 'output/features_multiscale/room_0'
    seq1_traj = 'dataset/room_0/Sequence_1/traj_w_c.txt'
    seq2_feature_dir = 'output/features_multiscale/room_0_seq2'
    seq2_traj = 'dataset/room_0/Sequence_2/traj_w_c.txt'
    cls_dir_seq1 = f'{seq1_feature_dir}/cls'
    cls_dir_seq2 = 'output/features_multiscale_compressed/room_0_seq2/cls'
    output_dir = Path('output/retrieval/room_0_vlad')
    output_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = 32  # AnyLoc 推荐值
    top_k = 5

    # ============================================================
    # 检查 Seq2 DINO patch tokens 是否存在
    # ============================================================
    seq2_dino_dir = Path(seq2_feature_dir) / 'fine_dino'
    if not seq2_dino_dir.exists():
        # Seq2 可能只有 CLS tokens (之前只提取了 CLS)
        # 需要判断是否有完整的 fine_dino 特征
        print(f"⚠️  Seq2 DINO patch 特征不存在: {seq2_dino_dir}")
        print(f"   之前只提取了 Seq2 的 CLS tokens。")
        print(f"   需要先为 Seq2 提取完整的 DINO patch 特征。")
        print(f"   运行: PYTHONPATH=. python scripts/extract_seq2_dino_patches.py")

        # 仍然可以评估 CLS token 方法
        print(f"\n先评估 CLS Token 基线...")
        has_vlad_query = False
    else:
        has_vlad_query = True

    # ============================================================
    # 1. 构建 VLAD 索引 (从 Seq1)
    # ============================================================
    print("\n" + "="*60)
    print("Step 1: 构建 VLAD 索引 (Seq1)")
    print("="*60)

    t0 = time.time()
    vlad_system = VLADPlaceRecognition.build_from_features(
        feature_dir=seq1_feature_dir,
        traj_path=seq1_traj,
        n_clusters=n_clusters,
        verbose=True,
    )
    print(f"  构建耗时: {time.time()-t0:.1f}s")

    # 保存
    vlad_system.save(str(output_dir))

    # ============================================================
    # 2. 加载 Seq2 查询数据
    # ============================================================
    print("\n" + "="*60)
    print("Step 2: 加载 Seq2 查询数据")
    print("="*60)

    # 位姿
    traj2 = np.loadtxt(seq2_traj)
    c2w_poses_2 = traj2.reshape(-1, 4, 4).astype(np.float32)
    w2c_poses_2 = np.linalg.inv(c2w_poses_2).astype(np.float32)

    # ============================================================
    # 3. CLS Token 基线评估
    # ============================================================
    print("\n" + "="*60)
    print("Step 3: CLS Token 基线评估")
    print("="*60)

    cls_seq2_dir = Path(cls_dir_seq2)
    if cls_seq2_dir.exists():
        # 构建 CLS 检索器
        cls_retriever = PlaceRecognition.build_from_dir(
            cls_dir=cls_dir_seq1,
            traj_path=seq1_traj,
        )

        # 加载 Seq2 CLS tokens
        cls_tokens_q = []
        q_frame_ids_cls = []
        q_poses_cls = []
        for fpath in sorted(cls_seq2_dir.glob('rgb_*_cls_*.pt')):
            import re
            match = re.search(r'rgb_(\d+)_cls_', fpath.name)
            if match:
                fid = int(match.group(1))
                if fid < len(w2c_poses_2):
                    t = torch.load(str(fpath), map_location='cpu').numpy().astype(np.float32)
                    cls_tokens_q.append(t)
                    q_frame_ids_cls.append(fid)
                    q_poses_cls.append(w2c_poses_2[fid])

        print(f"  Seq2 CLS tokens: {len(cls_tokens_q)}")

        cls_results = evaluate_retrieval(
            retriever=cls_retriever,
            query_features=cls_tokens_q,
            query_poses_w2c=q_poses_cls,
            query_frame_ids=q_frame_ids_cls,
            db_poses_w2c=cls_retriever.poses_w2c,
            db_frame_ids=cls_retriever.frame_ids,
            top_k=top_k,
            method_name="CLS Token (768d)",
            is_vlad=False,
        )
    else:
        print(f"  ⚠️ Seq2 CLS 目录不存在: {cls_seq2_dir}")
        cls_results = None

    # ============================================================
    # 4. VLAD 检索评估
    # ============================================================
    if has_vlad_query:
        print("\n" + "="*60)
        print("Step 4: VLAD 检索评估")
        print("="*60)

        # 加载 Seq2 DINO patch tokens
        vlad_tokens_q = []
        q_frame_ids_vlad = []
        q_poses_vlad = []

        for fpath in sorted(seq2_dino_dir.glob('rgb_*_fine_dino_*.pt')):
            import re
            match = re.search(r'rgb_(\d+)_fine_dino_', fpath.name)
            if match:
                fid = int(match.group(1))
                if fid < len(w2c_poses_2):
                    feat = torch.load(str(fpath), map_location='cpu').numpy()
                    vlad_tokens_q.append(feat)  # [768, H, W]
                    q_frame_ids_vlad.append(fid)
                    q_poses_vlad.append(w2c_poses_2[fid])

        print(f"  Seq2 DINO patch features: {len(vlad_tokens_q)}")

        vlad_results = evaluate_retrieval(
            retriever=vlad_system,
            query_features=vlad_tokens_q,
            query_poses_w2c=q_poses_vlad,
            query_frame_ids=q_frame_ids_vlad,
            db_poses_w2c=vlad_system.poses_w2c,
            db_frame_ids=vlad_system.frame_ids,
            top_k=top_k,
            method_name=f"VLAD (K={n_clusters}, {vlad_system.vlad_encoder.vlad_dim}d)",
            is_vlad=True,
        )
    else:
        vlad_results = None
        print("\n  ⚠️ 跳过 VLAD 评估 (Seq2 patch tokens 不存在)")
        print("  需要先为 Seq2 提取 DINO patch 特征")

    # ============================================================
    # 5. 对比总结
    # ============================================================
    print("\n" + "="*60)
    print("对比总结")
    print("="*60)

    comparison = {
        'cls': cls_results,
        'vlad': vlad_results,
    }

    if cls_results and vlad_results:
        print(f"\n{'方法':<35} {'Δt mean':>10} {'Δr mean':>10} {'Success%':>10}")
        print("-" * 65)
        for name, res in [("CLS Token (768d)", cls_results),
                          (f"VLAD (K={n_clusters})", vlad_results)]:
            if res and 'top1' in res:
                top1 = res['top1']
                print(f"{name:<35} {top1['mean_t']:>9.3f}m {top1['mean_r']:>9.1f}° "
                      f"{top1['success_rate']:>9.1f}%")

    # 保存结果
    with open(output_dir / 'eval_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n结果保存: {output_dir}")


if __name__ == '__main__':
    main()
