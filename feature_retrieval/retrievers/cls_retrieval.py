"""
Phase 3: DINO CLS Token Place Recognition (FAISS)
===================================================
使用 DINO CLS Token (768d) 构建 FAISS 索引，实现全局场景检索。

功能:
  1. 从压缩特征目录批量加载 CLS tokens → 构建 FAISS FlatIP 索引
  2. 给定查询 CLS token，检索 Top-K 最相似的训练帧
  3. 返回候选帧的 frame_id 和 W2C 位姿，作为迭代优化的初始估计

使用方式:
    # 构建索引
    retriever = PlaceRecognition.build_from_dir(
        cls_dir='output/features_multiscale_compressed/room_0/cls',
        traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
    )
    retriever.save('output/retrieval/room_0/index.faiss')

    # 加载索引 + 查询
    retriever = PlaceRecognition.load('output/retrieval/room_0/index.faiss')
    results = retriever.query(cls_token, top_k=5)
    # results[0] = {'frame_id': 42, 'score': 0.98, 'pose_w2c': [4,4]}
"""

import re
import json
import numpy as np
import torch
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union


class PlaceRecognition:
    """
    基于 DINO CLS Token 的全局场景检索。

    使用 FAISS IndexFlatIP (内积) 进行精确最近邻检索。
    因为 CLS token 已经 L2 归一化，内积 = 余弦相似度。
    """

    def __init__(
        self,
        dim: int = 768,
    ):
        self.dim = dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.frame_ids: List[int] = []
        self.poses_w2c: Optional[np.ndarray] = None  # [N, 4, 4]
        self.cls_tokens: Optional[np.ndarray] = None  # [N, dim]

    @classmethod
    def build_from_dir(
        cls,
        cls_dir: str,
        traj_path: str,
        dim: int = 768,
    ) -> 'PlaceRecognition':
        """
        从 CLS token 目录和位姿文件构建检索索引。

        Args:
            cls_dir: 包含 rgb_{id}_cls_768.pt 文件的目录
            traj_path: C2W 位姿文件 (traj_w_c.txt)
            dim: CLS token 维度

        Returns:
            构建好的 PlaceRecognition 实例
        """
        retriever = cls(dim=dim)

        # ── 加载位姿 (C2W → W2C) ──
        traj = np.loadtxt(traj_path)
        c2w_poses = traj.reshape(-1, 4, 4).astype(np.float32)
        all_w2c = np.linalg.inv(c2w_poses).astype(np.float32)
        total_poses = len(all_w2c)

        # ── 扫描 CLS token 文件 ──
        cls_path = Path(cls_dir)
        if not cls_path.exists():
            raise FileNotFoundError(f"CLS 目录不存在: {cls_path}")

        file_map: Dict[int, Path] = {}
        for fpath in cls_path.glob('rgb_*_cls_*.pt'):
            match = re.search(r'rgb_(\d+)_cls_', fpath.name)
            if match:
                fid = int(match.group(1))
                if fid < total_poses:
                    file_map[fid] = fpath

        sorted_fids = sorted(file_map.keys())
        n_frames = len(sorted_fids)

        if n_frames == 0:
            raise RuntimeError(f"在 {cls_path} 中未找到有效的 CLS token 文件")

        print(f"[PlaceRecognition] 扫描到 {n_frames} 个 CLS token")

        # ── 加载所有 CLS tokens ──
        tokens = np.zeros((n_frames, dim), dtype=np.float32)
        poses = np.zeros((n_frames, 4, 4), dtype=np.float32)

        for i, fid in enumerate(sorted_fids):
            t = torch.load(file_map[fid], map_location='cpu')
            tokens[i] = t.numpy().astype(np.float32)
            poses[i] = all_w2c[fid]

        # L2 归一化 (确保一致性，虽然提取时已归一化)
        faiss.normalize_L2(tokens)

        # ── 构建 FAISS 索引 ──
        index = faiss.IndexFlatIP(dim)  # 内积 = cosine similarity (已归一化)
        index.add(tokens)

        retriever.index = index
        retriever.frame_ids = sorted_fids
        retriever.poses_w2c = poses
        retriever.cls_tokens = tokens

        print(f"[PlaceRecognition] FAISS 索引构建完成: {index.ntotal} vectors, {dim}d")
        return retriever

    def query(
        self,
        cls_token: Union[torch.Tensor, np.ndarray],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        检索与查询 CLS token 最相似的 Top-K 训练帧。

        Args:
            cls_token: 查询向量 [768] 或 [1, 768]
            top_k: 返回的候选数量

        Returns:
            列表，每项包含:
              - frame_id: int
              - score: float (余弦相似度)
              - pose_w2c: np.ndarray [4, 4]
        """
        if self.index is None:
            raise RuntimeError("索引未构建，请先调用 build_from_dir() 或 load()")

        # 转换为 numpy
        if isinstance(cls_token, torch.Tensor):
            q = cls_token.detach().cpu().numpy().astype(np.float32)
        else:
            q = cls_token.astype(np.float32)

        q = q.reshape(1, -1)
        faiss.normalize_L2(q)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, top_k)

        results = []
        for rank in range(top_k):
            idx = int(indices[0, rank])
            results.append({
                'frame_id': self.frame_ids[idx],
                'score': float(scores[0, rank]),
                'pose_w2c': self.poses_w2c[idx].copy(),
                'rank': rank,
            })

        return results

    def query_batch(
        self,
        cls_tokens: Union[torch.Tensor, np.ndarray],
        top_k: int = 5,
    ) -> List[List[Dict]]:
        """
        批量检索。

        Args:
            cls_tokens: [B, 768]
            top_k: 每个查询返回的候选数

        Returns:
            长度为 B 的列表，每项是 top_k 个候选的列表
        """
        if self.index is None:
            raise RuntimeError("索引未构建")

        if isinstance(cls_tokens, torch.Tensor):
            q = cls_tokens.detach().cpu().numpy().astype(np.float32)
        else:
            q = cls_tokens.astype(np.float32)

        q = q.reshape(-1, self.dim)
        faiss.normalize_L2(q)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, top_k)

        batch_results = []
        for b in range(len(q)):
            results = []
            for rank in range(top_k):
                idx = int(indices[b, rank])
                results.append({
                    'frame_id': self.frame_ids[idx],
                    'score': float(scores[b, rank]),
                    'pose_w2c': self.poses_w2c[idx].copy(),
                    'rank': rank,
                })
            batch_results.append(results)

        return batch_results

    def save(self, save_path: str):
        """
        保存索引 + 元数据到磁盘。

        保存结构:
          {save_path}         ← FAISS index
          {save_path}.meta    ← frame_ids + poses (npz)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        faiss.write_index(self.index, str(save_path))

        # 保存元数据
        meta_path = save_path.with_suffix('.meta.npz')
        np.savez(
            meta_path,
            frame_ids=np.array(self.frame_ids, dtype=np.int64),
            poses_w2c=self.poses_w2c,
            cls_tokens=self.cls_tokens,
            dim=np.array([self.dim]),
        )

        print(f"[PlaceRecognition] 已保存: {save_path}")
        print(f"  索引: {self.index.ntotal} vectors")
        print(f"  元数据: {meta_path}")

    @classmethod
    def load(cls, load_path: str) -> 'PlaceRecognition':
        """
        从磁盘加载预构建的索引。

        Args:
            load_path: FAISS index 文件路径

        Returns:
            加载好的 PlaceRecognition 实例
        """
        load_path = Path(load_path)
        meta_path = load_path.with_suffix('.meta.npz')

        if not load_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {load_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {meta_path}")

        # 加载 FAISS 索引
        index = faiss.read_index(str(load_path))

        # 加载元数据
        meta = np.load(meta_path, allow_pickle=False)
        dim = int(meta['dim'][0])

        retriever = cls(dim=dim)
        retriever.index = index
        retriever.frame_ids = meta['frame_ids'].tolist()
        retriever.poses_w2c = meta['poses_w2c'].astype(np.float32)
        retriever.cls_tokens = meta['cls_tokens'].astype(np.float32)

        print(f"[PlaceRecognition] 已加载: {load_path}")
        print(f"  索引: {index.ntotal} vectors, {dim}d")
        return retriever

    def get_pose(self, frame_id: int) -> Optional[np.ndarray]:
        """获取指定帧的 W2C 位姿。"""
        if frame_id in self.frame_ids:
            idx = self.frame_ids.index(frame_id)
            return self.poses_w2c[idx].copy()
        return None

    def get_cls_token(self, frame_id: int) -> Optional[np.ndarray]:
        """获取指定帧的 CLS token。"""
        if frame_id in self.frame_ids:
            idx = self.frame_ids.index(frame_id)
            return self.cls_tokens[idx].copy()
        return None

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __repr__(self) -> str:
        n = self.index.ntotal if self.index else 0
        return f"PlaceRecognition(dim={self.dim}, n_frames={n})"
