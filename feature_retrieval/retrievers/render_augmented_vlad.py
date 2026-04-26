#!/usr/bin/env python3
"""
3DGS 渲染增强 VLAD 检索 Pipeline
==================================

通过 3DGS 渲染多视角 DINO patch features，构建视角覆盖更密的检索数据库。

核心流程:
  1. 从 Seq1 轨迹中稀疏采样种子帧 (覆盖全场景)
  2. 对每个种子帧生成多个视角变体 (旋转扰动)
  3. 加载 fine_dino 3DGS 模型，渲染每个新位姿的 patch feature map
  4. 用 VLAD 编码器对渲染特征计算描述子
  5. 构建增强 FAISS 索引

使用示例:
    python -m feature_retrieval.retrievers.render_augmented_vlad \
      --ply_path dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply \
      --feature_ckpt output/feature_3dgs/room_0_raw/fine_dino/best_model.pth \
      --traj_path dataset/room_0/Sequence_1/traj_w_c.txt \
      --output_dir output/retrieval/room_0_augmented \
      --seed_step 10 --n_yaw 8 --n_pitch 3 \
      --vlad_clusters 32

评估:
    python -m feature_retrieval.retrievers.render_augmented_vlad \
      --eval \
      --index_dir output/retrieval/room_0_augmented \
      --query_feature_dir output/features_multiscale/room_0_seq2/fine_dino \
      --query_traj_path dataset/room_0/Sequence_2/traj_w_c.txt
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import faiss

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_gaussian.models.gaussian_feature_model import GaussianFeatureModel
from feature_gaussian.models.raw_gaussian_model import RawScaleGaussianModel
from feature_retrieval.retrievers.vlad_retrieval import VLADEncoder, VLADPlaceRecognition
from feature_field.models.feature_renderer import FeatureRenderer


# ============================================================
# 1. 种子帧稀疏采样
# ============================================================

def select_seed_frames(
    c2w_poses: np.ndarray,
    method: str = 'uniform',
    step: int = 10,
    min_distance: float = 0.3,
) -> List[int]:
    """
    从训练轨迹中选择种子帧 (稀疏且覆盖全场景)。

    Args:
        c2w_poses: [N, 4, 4] C2W 位姿
        method: 'uniform' (等间隔采样) / 'farthest' (最远点采样, 覆盖更均匀)
        step: uniform 模式的采样步长
        min_distance: farthest 模式的最小间距 (m)

    Returns:
        种子帧 ID 列表
    """
    N = len(c2w_poses)
    positions = c2w_poses[:, :3, 3]  # [N, 3]

    if method == 'uniform':
        seeds = list(range(0, N, step))
    elif method == 'farthest':
        # 最远点采样: 保证空间覆盖均匀
        seeds = [0]
        selected_pos = [positions[0]]
        for _ in range(N):
            # 计算所有点到已选点的最小距离
            dists_to_selected = np.array([
                np.linalg.norm(positions - sp, axis=1)
                for sp in selected_pos
            ])  # [n_selected, N]
            min_dists = dists_to_selected.min(axis=0)  # [N]

            # 选最远的点
            next_idx = np.argmax(min_dists)
            if min_dists[next_idx] < min_distance:
                break  # 所有点都已被覆盖
            seeds.append(int(next_idx))
            selected_pos.append(positions[next_idx])
    else:
        raise ValueError(f"Unknown method: {method}")

    return sorted(seeds)


# ============================================================
# 2. 多视角位姿生成
# ============================================================

def generate_augmented_poses(
    c2w: np.ndarray,
    n_yaw: int = 8,
    n_pitch: int = 3,
    yaw_range: float = 180.0,
    pitch_range: float = 30.0,
) -> List[np.ndarray]:
    """
    围绕种子位姿生成多个视角变体 (绕相机中心旋转)。

    Args:
        c2w: [4, 4] 种子帧的 C2W 位姿
        n_yaw: yaw (水平旋转) 的采样数
        n_pitch: pitch (俯仰) 的采样数
        yaw_range: yaw 覆盖范围 (°), 360° = 全向
        pitch_range: pitch 覆盖范围 (°)

    Returns:
        变体位姿列表 (C2W), 包含原始位姿
    """
    augmented = [c2w.copy()]  # 包含原始位姿

    # Yaw 角度列表 (均匀分布, 排除 0°)
    if n_yaw > 0:
        yaw_angles = np.linspace(-yaw_range, yaw_range, n_yaw + 1)[:-1]
        yaw_angles = yaw_angles[yaw_angles != 0]  # 排除原始方向
    else:
        yaw_angles = []

    # Pitch 角度列表
    if n_pitch > 1:
        pitch_angles = np.linspace(-pitch_range, pitch_range, n_pitch)
        pitch_angles = pitch_angles[pitch_angles != 0]  # 排除 0°
    else:
        pitch_angles = []

    position = c2w[:3, 3].copy()
    R_orig = c2w[:3, :3].copy()

    for yaw_deg in yaw_angles:
        yaw_rad = np.radians(yaw_deg)
        # 绕世界坐标系 Y 轴旋转 (yaw)
        R_yaw = np.array([
            [np.cos(yaw_rad),  0, np.sin(yaw_rad)],
            [0,                 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)],
        ], dtype=np.float32)

        R_new = R_yaw @ R_orig
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R_new
        pose[:3, 3] = position
        augmented.append(pose)

    # Pitch 扰动 (在原始方向上叠加)
    for pitch_deg in pitch_angles:
        pitch_rad = np.radians(pitch_deg)
        # 绕相机的 X 轴旋转 (pitch, 相机局部坐标)
        R_pitch = np.array([
            [1, 0,                  0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad),  np.cos(pitch_rad)],
        ], dtype=np.float32)

        # 组合: 先在相机局部坐标旋转, 再转回世界坐标
        R_new = R_orig @ R_pitch
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R_new
        pose[:3, 3] = position
        augmented.append(pose)

    # Yaw + Pitch 组合 (只取少量)
    for yaw_deg in yaw_angles[::max(1, len(yaw_angles) // 4)]:
        for pitch_deg in pitch_angles:
            yaw_rad = np.radians(yaw_deg)
            pitch_rad = np.radians(pitch_deg)
            R_yaw = np.array([
                [np.cos(yaw_rad),  0, np.sin(yaw_rad)],
                [0,                 1, 0],
                [-np.sin(yaw_rad), 0, np.cos(yaw_rad)],
            ], dtype=np.float32)
            R_pitch = np.array([
                [1, 0,                  0],
                [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                [0, np.sin(pitch_rad),  np.cos(pitch_rad)],
            ], dtype=np.float32)
            R_new = R_yaw @ R_orig @ R_pitch
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R_new
            pose[:3, 3] = position
            augmented.append(pose)

    return augmented


# ============================================================
# 3. 加载 3DGS 模型
# ============================================================

def load_dino_3dgs(
    ply_path: str,
    ckpt_path: str,
    device: torch.device,
) -> RawScaleGaussianModel:
    """
    加载 fine_dino 3DGS 模型 (几何 from PLY + 特征 from checkpoint)。

    Args:
        ply_path: 原始 3DGS PLY 路径
        ckpt_path: best_model.pth 路径
        device: GPU device

    Returns:
        加载完毕的 RawScaleGaussianModel
    """
    model = RawScaleGaussianModel(scale='fine_dino')
    model.load_ply(ply_path)
    model = model.to(device)

    # 恢复训练好的特征嵌入
    ckpt = torch.load(ckpt_path, map_location=device)
    model._loc_feature.data.copy_(ckpt['loc_feature'])

    feat_dim = ckpt['feature_dim']
    iteration = ckpt['iteration']
    loss = ckpt['loss']
    print(f"[load_dino_3dgs] 恢复特征: dim={feat_dim}, iter={iteration}, loss={loss:.6f}")

    return model


# ============================================================
# 4. 批量渲染 + VLAD 编码
# ============================================================

def render_and_encode(
    model: GaussianFeatureModel,
    poses_c2w: List[np.ndarray],
    vlad_encoder: VLADEncoder,
    feat_h: int = 35,
    feat_w: int = 46,
    img_h: int = 480,
    img_w: int = 640,
    fx: float = 320.0,
    fy: float = 320.0,
    cx: float = 319.5,
    cy: float = 239.5,
    device: torch.device = None,
    batch_log_interval: int = 50,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    对一组位姿渲染 DINO 特征并计算 VLAD 描述子。

    Args:
        model: 加载好特征的 3DGS 模型
        poses_c2w: C2W 位姿列表
        vlad_encoder: 已 fit 的 VLAD 编码器
        feat_h, feat_w: 特征图分辨率
        img_h, img_w: 原图分辨率 (用于 gsplat 投影)
        fx, fy, cx, cy: 相机内参

    Returns:
        (vlad_descriptors [N, vlad_dim], w2c_poses [N, 4, 4])
    """
    if device is None:
        device = model.get_xyz.device

    N = len(poses_c2w)
    descriptors = []
    w2c_list = []

    t_start = time.time()

    with torch.no_grad():
        for i, c2w in enumerate(poses_c2w):
            w2c = np.linalg.inv(c2w).astype(np.float32)
            viewmat = torch.from_numpy(w2c).to(device)

            # 渲染 DINO 特征图
            result = FeatureRenderer.render_features(
                gaussian_model=model,
                viewmat=viewmat,
                fx=fx, fy=fy,
                cx=cx, cy=cy,
                img_height=img_h,
                img_width=img_w,
                feature_height=feat_h,
                feature_width=feat_w,
                norm_feat_before_render=True,
                norm_feat_after_render=True,
            )

            feat_map = result['feature_map']  # [768, 35, 46]

            # 转为 patch tokens: [768, 35, 46] → [35*46, 768]
            D, H, W = feat_map.shape
            patches = feat_map.reshape(D, -1).T.cpu().numpy()  # [1610, 768]

            # VLAD 编码
            vlad = vlad_encoder.encode_single(patches)
            descriptors.append(vlad)
            w2c_list.append(w2c)

            if (i + 1) % batch_log_interval == 0 or (i + 1) == N:
                elapsed = time.time() - t_start
                fps = (i + 1) / elapsed
                print(f"  渲染进度: {i+1}/{N} ({fps:.1f} frames/s)")

    descriptors = np.stack(descriptors, axis=0)  # [N, vlad_dim]
    w2c_array = np.stack(w2c_list, axis=0)  # [N, 4, 4]

    return descriptors, w2c_array


# ============================================================
# 5. 主流程: 构建增强数据库
# ============================================================

def load_real_dino_patches(
    feature_dir: str,
    traj_path: str,
    frame_ids: Optional[List[int]] = None,
) -> Tuple[List[int], List[np.ndarray], np.ndarray]:
    """
    加载真实 DINO patch tokens (从文件)。

    Returns:
        (frame_ids, patch_tokens_list, w2c_poses)
    """
    feature_dir = Path(feature_dir)
    dino_dir = feature_dir / 'fine_dino' if (feature_dir / 'fine_dino').exists() else feature_dir

    traj = np.loadtxt(traj_path)
    c2w_all = traj.reshape(-1, 4, 4).astype(np.float32)
    w2c_all = np.linalg.inv(c2w_all).astype(np.float32)

    file_map = {}
    for fpath in dino_dir.glob('rgb_*_fine_dino_*.pt'):
        match = re.search(r'rgb_(\d+)_fine_dino_', fpath.name)
        if match:
            fid = int(match.group(1))
            if fid < len(w2c_all):
                file_map[fid] = fpath

    if frame_ids is None:
        frame_ids = sorted(file_map.keys())
    else:
        frame_ids = [fid for fid in frame_ids if fid in file_map]

    patches_list = []
    valid_fids = []
    for fid in frame_ids:
        feat = torch.load(str(file_map[fid]), map_location='cpu').numpy()
        D, H, W = feat.shape
        patches = feat.reshape(D, -1).T  # [H*W, D]
        patches_list.append(patches)
        valid_fids.append(fid)

    w2c_poses = np.stack([w2c_all[fid] for fid in valid_fids])
    return valid_fids, patches_list, w2c_poses


def build_augmented_database(args):
    """
    构建混合数据库: 真实 DINO 特征 (全部原始帧) + 3DGS 渲染增强 (新视角)。

    关键设计:
      - K-means 在真实 patch tokens 上训练 (避免域差异)
      - 原始帧使用真实特征 (不用渲染)
      - 增强帧使用 3DGS 渲染特征 (补充视角)
    """
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("混合数据库: 真实特征 + 3DGS 渲染增强")
    print("=" * 65)

    # ── 1. 加载轨迹 ──
    print("\n[Step 1] 加载 Seq1 轨迹...")
    traj = np.loadtxt(args.traj_path)
    c2w_all = traj.reshape(-1, 4, 4).astype(np.float32)
    print(f"  总帧数: {len(c2w_all)}")

    # ── 2. 加载真实 DINO patch tokens (全部原始帧) ──
    print("\n[Step 2] 加载真实 DINO patch tokens...")
    real_fids, real_patches, real_w2c = load_real_dino_patches(
        args.real_feature_dir, args.traj_path
    )
    n_real = len(real_fids)
    print(f"  真实帧: {n_real}")

    # ── 3. 训练 VLAD 编码器 (在真实特征上!) ──
    print(f"\n[Step 3] 训练 VLAD 编码器 (K={args.vlad_clusters}, 真实数据)...")
    all_real_patches = np.concatenate(real_patches, axis=0)  # [N_total, 768]
    print(f"  真实 patch tokens: {all_real_patches.shape[0]:,} × {all_real_patches.shape[1]}d")

    vlad_encoder = VLADEncoder(
        n_clusters=args.vlad_clusters,
        token_dim=all_real_patches.shape[1],
    )
    vlad_encoder.fit(all_real_patches, max_samples=args.vlad_max_kmeans, verbose=True)
    del all_real_patches  # 释放内存

    # ── 4. 编码真实帧 ──
    print(f"\n[Step 4] 编码真实帧 VLAD ({n_real} 帧)...")
    real_descriptors = vlad_encoder.encode_batch(real_patches)
    print(f"  VLAD 描述子: {real_descriptors.shape}")

    # ── 5. 选择种子帧用于视角增强 ──
    print(f"\n[Step 5] 选择种子帧 (method={args.seed_method})...")
    seed_ids = select_seed_frames(
        c2w_all,
        method=args.seed_method,
        step=args.seed_step,
        min_distance=args.min_seed_distance,
    )
    print(f"  种子帧: {len(seed_ids)} 帧")

    # ── 6. 生成增强位姿 (只生成新视角, 不含种子帧原始位姿) ──
    print(f"\n[Step 6] 生成视角变体 (yaw={args.n_yaw}, pitch={args.n_pitch})...")
    aug_poses_c2w = []
    aug_source_ids = []

    for seed_id in seed_ids:
        c2w_seed = c2w_all[seed_id]
        augmented = generate_augmented_poses(
            c2w_seed,
            n_yaw=args.n_yaw,
            n_pitch=args.n_pitch,
            yaw_range=args.yaw_range,
            pitch_range=args.pitch_range,
        )
        # 跳过第一个 (原始位姿, 已在真实帧数据库中)
        for pose in augmented[1:]:
            aug_poses_c2w.append(pose)
            aug_source_ids.append(seed_id)

    n_aug = len(aug_poses_c2w)
    print(f"  渲染增强帧: {n_aug}")

    # ── 7. 加载 3DGS 模型并渲染增强帧 ──
    print("\n[Step 7] 加载 3DGS 模型并渲染增强帧...")
    model = load_dino_3dgs(args.ply_path, args.feature_ckpt, device)

    aug_descriptors, aug_w2c = render_and_encode(
        model=model,
        poses_c2w=aug_poses_c2w,
        vlad_encoder=vlad_encoder,
        feat_h=35, feat_w=46,
        img_h=args.img_height, img_w=args.img_width,
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
        device=device,
        batch_log_interval=100,
    )

    # 释放 GPU 内存
    del model
    torch.cuda.empty_cache()

    # ── 8. 合并数据库 ──
    print("\n[Step 8] 合并数据库...")
    all_descriptors = np.concatenate([real_descriptors, aug_descriptors], axis=0)
    all_w2c = np.concatenate([real_w2c, aug_w2c], axis=0)
    all_is_original = np.array(
        [True] * n_real + [False] * n_aug
    )
    all_source_ids = np.array(
        list(real_fids) + aug_source_ids
    )

    n_total = len(all_descriptors)
    print(f"  总数据库: {n_total} (真实: {n_real} + 渲染: {n_aug})")

    # ── 9. 构建 FAISS 索引 ──
    print("\n[Step 9] 构建 FAISS 索引...")
    vlad_dim = all_descriptors.shape[1]
    faiss.normalize_L2(all_descriptors)
    index = faiss.IndexFlatIP(vlad_dim)
    index.add(all_descriptors)
    print(f"  FAISS 索引: {index.ntotal} vectors, {vlad_dim}d")

    # ── 10. 保存 ──
    print("\n[Step 10] 保存数据库...")
    vlad_encoder.save(str(output_dir / 'vlad_encoder'))
    faiss.write_index(index, str(output_dir / 'augmented_index.faiss'))
    np.save(output_dir / 'descriptors.npy', all_descriptors)
    np.save(output_dir / 'poses_w2c.npy', all_w2c)
    np.save(output_dir / 'source_frame_ids.npy', all_source_ids)
    np.save(output_dir / 'is_original.npy', all_is_original)

    meta = {
        'n_total': n_total,
        'n_real': n_real,
        'n_augmented': n_aug,
        'n_seeds': len(seed_ids),
        'seed_ids': seed_ids,
        'seed_method': args.seed_method,
        'seed_step': args.seed_step,
        'n_yaw': args.n_yaw,
        'n_pitch': args.n_pitch,
        'yaw_range': args.yaw_range,
        'pitch_range': args.pitch_range,
        'vlad_clusters': args.vlad_clusters,
        'vlad_dim': vlad_dim,
        'feature_ckpt': args.feature_ckpt,
        'ply_path': args.ply_path,
    }
    with open(output_dir / 'database_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"混合数据库构建完成!")
    print(f"  真实帧: {n_real} + 渲染增强: {n_aug} = 总计: {n_total}")
    print(f"  VLAD 维度: {vlad_dim}d (K={args.vlad_clusters})")
    print(f"  保存: {output_dir}")
    print(f"{'='*65}")


# ============================================================
# 6. 评估: Seq2 跨序列检索
# ============================================================

def evaluate_cross_sequence(args):
    """用 Seq2 的真实 DINO patch tokens 查询增强数据库。"""
    print("=" * 65)
    print("跨序列检索评估: Seq2 → 增强数据库")
    print("=" * 65)

    index_dir = Path(args.index_dir)

    # ── 加载数据库 ──
    print("\n[1] 加载增强数据库...")
    vlad_encoder = VLADEncoder.load(str(index_dir / 'vlad_encoder'))
    index = faiss.read_index(str(index_dir / 'augmented_index.faiss'))
    poses_w2c = np.load(index_dir / 'poses_w2c.npy')
    is_original = np.load(index_dir / 'is_original.npy')

    with open(index_dir / 'database_meta.json', 'r') as f:
        meta = json.load(f)

    n_real = meta.get('n_real', meta.get('n_original', 0))
    n_aug = meta.get('n_augmented', meta.get('n_rendered', 0))
    print(f"  数据库: {index.ntotal} 帧 (真实: {n_real}, 增强: {n_aug})")

    # ── 加载 Seq2 GT ──
    print("\n[2] 加载 Seq2...")
    query_traj = np.loadtxt(args.query_traj_path)
    c2w_query = query_traj.reshape(-1, 4, 4).astype(np.float32)
    w2c_query = np.linalg.inv(c2w_query).astype(np.float32)

    # ── 加载 Seq2 DINO patch tokens ──
    query_dir = Path(args.query_feature_dir)
    query_data = []

    for fpath in sorted(query_dir.glob('rgb_*_fine_dino_*.pt')):
        match = re.search(r'rgb_(\d+)_fine_dino_', fpath.name)
        if match:
            fid = int(match.group(1))
            if fid < len(w2c_query):
                feat = torch.load(str(fpath), map_location='cpu').numpy()
                # [768, 35, 46] → [1610, 768]
                patches = feat.reshape(feat.shape[0], -1).T
                query_data.append((fid, patches, w2c_query[fid]))

    print(f"  Seq2 帧: {len(query_data)}")

    # ── 批量检索 ──
    print("\n[3] 批量检索...")
    trans_errors = []
    rot_errors = []

    for i, (fid, patches, gt_w2c) in enumerate(query_data):
        vlad = vlad_encoder.encode_single(patches.astype(np.float32)).reshape(1, -1)
        faiss.normalize_L2(vlad)

        scores, indices = index.search(vlad, 1)
        best_idx = int(indices[0, 0])
        retrieved_w2c = poses_w2c[best_idx]

        # 平移误差
        t_err = np.linalg.norm(gt_w2c[:3, 3] - retrieved_w2c[:3, 3])
        trans_errors.append(t_err)

        # 旋转误差
        R_rel = gt_w2c[:3, :3] @ retrieved_w2c[:3, :3].T
        trace = np.clip(np.trace(R_rel), -1.0, 3.0)
        angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        rot_errors.append(np.degrees(angle))

    trans_arr = np.array(trans_errors)
    rot_arr = np.array(rot_errors)

    # ── 结果输出 ──
    print(f"\n{'='*65}")
    print(f"检索结果 ({len(query_data)} 查询 → {index.ntotal} 数据库)")
    print(f"{'='*65}")
    print(f"  平移误差: mean={trans_arr.mean():.4f}m  median={np.median(trans_arr):.4f}m  "
          f"std={trans_arr.std():.4f}")
    print(f"  旋转误差: mean={rot_arr.mean():.2f}°  median={np.median(rot_arr):.2f}°  "
          f"std={rot_arr.std():.2f}")

    thresholds = [
        (0.05, 5), (0.1, 10), (0.25, 15),
        (0.5, 30), (1.0, 45),
    ]
    for t_th, r_th in thresholds:
        pct = np.mean((trans_arr < t_th) & (rot_arr < r_th)) * 100
        print(f"  < {t_th}m & {r_th}°: {pct:.1f}%")

    # 保存结果
    results = {
        'n_queries': len(query_data),
        'n_database': index.ntotal,
        'trans_mean': float(trans_arr.mean()),
        'trans_median': float(np.median(trans_arr)),
        'rot_mean': float(rot_arr.mean()),
        'rot_median': float(np.median(rot_arr)),
    }
    for t_th, r_th in thresholds:
        key = f'recall_{t_th}m_{r_th}deg'
        results[key] = float(np.mean((trans_arr < t_th) & (rot_arr < r_th)) * 100)

    result_path = index_dir / 'eval_results.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  结果保存: {result_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='3DGS 渲染增强 VLAD 检索')
    sub = parser.add_subparsers(dest='command')

    # ── build: 构建增强数据库 ──
    build_p = sub.add_parser('build', help='构建增强数据库')
    build_p.add_argument('--ply_path', required=True, help='原始 3DGS PLY 路径')
    build_p.add_argument('--feature_ckpt', required=True, help='fine_dino best_model.pth')
    build_p.add_argument('--traj_path', required=True, help='Seq1 C2W 轨迹')
    build_p.add_argument('--real_feature_dir', required=True, help='Seq1 真实 DINO 特征目录')
    build_p.add_argument('--output_dir', required=True, help='输出目录')

    # 种子帧采样
    build_p.add_argument('--seed_method', default='farthest', choices=['uniform', 'farthest'])
    build_p.add_argument('--seed_step', type=int, default=10, help='uniform 采样步长')
    build_p.add_argument('--min_seed_distance', type=float, default=0.3, help='farthest 最小间距 (m)')

    # 视角生成
    build_p.add_argument('--n_yaw', type=int, default=8, help='水平旋转采样数')
    build_p.add_argument('--n_pitch', type=int, default=3, help='俯仰采样数')
    build_p.add_argument('--yaw_range', type=float, default=90.0, help='yaw 覆盖范围 (°)')
    build_p.add_argument('--pitch_range', type=float, default=20.0, help='pitch 覆盖范围 (°)')

    # VLAD
    build_p.add_argument('--vlad_clusters', type=int, default=32, help='VLAD 聚类数')
    build_p.add_argument('--vlad_max_kmeans', type=int, default=100000, help='K-means 最大样本')

    # 相机参数
    build_p.add_argument('--img_height', type=int, default=480)
    build_p.add_argument('--img_width', type=int, default=640)
    build_p.add_argument('--fx', type=float, default=320.0)
    build_p.add_argument('--fy', type=float, default=320.0)
    build_p.add_argument('--cx', type=float, default=319.5)
    build_p.add_argument('--cy', type=float, default=239.5)

    # ── eval: 评估 ──
    eval_p = sub.add_parser('eval', help='跨序列检索评估')
    eval_p.add_argument('--index_dir', required=True, help='增强数据库目录')
    eval_p.add_argument('--query_feature_dir', required=True, help='Seq2 fine_dino 特征目录')
    eval_p.add_argument('--query_traj_path', required=True, help='Seq2 C2W 轨迹')

    args = parser.parse_args()

    if args.command == 'build':
        build_augmented_database(args)
    elif args.command == 'eval':
        evaluate_cross_sequence(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
