"""
AnyLoc-Style VLAD Place Recognition
=====================================
基于 DINOv2 patch tokens + VLAD 聚合的通用场景检索。

核心思路 (AnyLoc: Towards Universal Visual Place Recognition):
  1. 使用 DINOv2 提取 patch-level 特征 (已有: fine_dino 768d @ 35×46 = 1610 patches/image)
  2. 从训练集 patch tokens 学习 K-means 聚类中心 (K=32~64)
  3. 对每帧图像计算 VLAD 描述子:
     - 将每个 patch token 分配到最近聚类
     - 对每个聚类，计算残差之和 (token - center)
     - K 个残差向量拼接 → K×D 维全局描述子
     - Intra-normalize + L2 normalize
  4. 使用 FAISS IndexFlatIP 进行余弦相似度检索

与 CLS Token 的区别:
  - CLS: 768d 全局语义，忽略空间结构
  - VLAD: K×768d 空间-语义聚合，保留局部结构信息

与 NetVLAD 的区别:
  - NetVLAD: 需要 VGG16 backbone + 端到端训练 (Pitts30K)
  - AnyLoc VLAD: 直接用预训练 DINOv2 tokens + 免训练 K-means, 通用性更强

参考:
  - AnyLoc: https://arxiv.org/abs/2308.00688
  - VLAD: Jégou et al., "Aggregating local descriptors into a compact image representation"
"""

import numpy as np
import torch
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import re
import json
from sklearn.cluster import MiniBatchKMeans


class VLADEncoder:
    """
    VLAD (Vector of Locally Aggregated Descriptors) 编码器。

    将 patch-level tokens 聚合为紧凑的全局描述子。
    """

    def __init__(
        self,
        n_clusters: int = 32,
        token_dim: int = 768,
        pca_dim: Optional[int] = None,
    ):
        """
        Args:
            n_clusters: VLAD 聚类数 (AnyLoc 推荐 32)
            token_dim: Patch token 维度 (DINOv2 = 768)
            pca_dim: 可选 PCA 降维 (None = 不降维, 输出 K×D 维)
        """
        self.n_clusters = n_clusters
        self.token_dim = token_dim
        self.pca_dim = pca_dim
        self.vlad_dim = n_clusters * token_dim  # 32×768 = 24576

        self.cluster_centers: Optional[np.ndarray] = None  # [K, D]
        self.kmeans: Optional[MiniBatchKMeans] = None
        self.pca_transform: Optional[np.ndarray] = None  # [pca_dim, vlad_dim]
        self.pca_mean: Optional[np.ndarray] = None

    def fit(
        self,
        patch_tokens: np.ndarray,
        max_samples: int = 100000,
        verbose: bool = True,
    ):
        """
        从训练集 patch tokens 学习聚类中心。

        Args:
            patch_tokens: [N_total, D] 所有训练帧的 patch tokens 展平
            max_samples: K-means 最大采样数 (内存/速度考虑)
            verbose: 打印进度
        """
        N, D = patch_tokens.shape
        assert D == self.token_dim, f"Token dim mismatch: {D} vs {self.token_dim}"

        if verbose:
            print(f"[VLADEncoder] 学习聚类中心...")
            print(f"  Patch tokens: {N:,} × {D}d")
            print(f"  K-means clusters: {self.n_clusters}")

        # 采样 (如果 token 数量过大)
        if N > max_samples:
            indices = np.random.choice(N, max_samples, replace=False)
            tokens_sample = patch_tokens[indices]
            if verbose:
                print(f"  采样: {max_samples:,} / {N:,}")
        else:
            tokens_sample = patch_tokens

        # K-Means 聚类
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=min(4096, len(tokens_sample)),
            n_init=3,
            max_iter=100,
            random_state=42,
            verbose=0,
        )
        self.kmeans.fit(tokens_sample)
        self.cluster_centers = self.kmeans.cluster_centers_.astype(np.float32)  # [K, D]

        if verbose:
            print(f"  聚类完成: {self.cluster_centers.shape}")

    def encode_single(self, patch_tokens: np.ndarray) -> np.ndarray:
        """
        对单帧 patch tokens 计算 VLAD 描述子。

        Args:
            patch_tokens: [n_patches, D] 单帧所有 patch tokens

        Returns:
            vlad: [K*D] VLAD 描述子 (已 L2 归一化)
        """
        assert self.cluster_centers is not None, "请先调用 fit()"

        n_patches, D = patch_tokens.shape
        K = self.n_clusters

        # 1. 分配 patch tokens 到最近聚类
        assignments = self.kmeans.predict(patch_tokens)  # [n_patches]

        # 2. 计算每个聚类的残差之和
        vlad = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            mask = (assignments == k)
            if mask.any():
                residuals = patch_tokens[mask] - self.cluster_centers[k]  # [n_k, D]
                vlad[k] = residuals.sum(axis=0)

        # 3. Intra-normalize (per-cluster L2 normalize, AnyLoc 关键步骤)
        for k in range(K):
            norm = np.linalg.norm(vlad[k])
            if norm > 1e-8:
                vlad[k] /= norm

        # 4. Flatten + Global L2 normalize
        vlad_flat = vlad.reshape(-1)  # [K*D]
        global_norm = np.linalg.norm(vlad_flat)
        if global_norm > 1e-8:
            vlad_flat /= global_norm

        return vlad_flat

    def encode_batch(self, patch_tokens_list: List[np.ndarray]) -> np.ndarray:
        """
        批量编码 VLAD 描述子。

        Args:
            patch_tokens_list: 列表，每项 [n_patches, D]

        Returns:
            vlad_descriptors: [B, K*D]
        """
        descriptors = []
        for tokens in patch_tokens_list:
            descriptors.append(self.encode_single(tokens))
        return np.stack(descriptors, axis=0)

    def save(self, save_dir: str):
        """保存聚类中心和配置。"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / 'cluster_centers.npy', self.cluster_centers)
        config = {
            'n_clusters': self.n_clusters,
            'token_dim': self.token_dim,
            'vlad_dim': self.vlad_dim,
            'pca_dim': self.pca_dim,
        }
        with open(save_dir / 'vlad_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[VLADEncoder] 保存: {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> 'VLADEncoder':
        """加载已保存的 VLAD 编码器。"""
        save_dir = Path(save_dir)
        with open(save_dir / 'vlad_config.json', 'r') as f:
            config = json.load(f)

        encoder = cls(
            n_clusters=config['n_clusters'],
            token_dim=config['token_dim'],
            pca_dim=config.get('pca_dim'),
        )
        encoder.cluster_centers = np.load(save_dir / 'cluster_centers.npy')

        # 重建 KMeans (只需要 predict)
        encoder.kmeans = MiniBatchKMeans(n_clusters=encoder.n_clusters)
        encoder.kmeans.cluster_centers_ = encoder.cluster_centers
        # 标记为已拟合 (sklearn hack)
        encoder.kmeans._n_threads = 1
        encoder.kmeans.n_features_in_ = encoder.token_dim
        encoder.kmeans.labels_ = np.zeros(0)
        encoder.kmeans._n_init = 1

        print(f"[VLADEncoder] 加载: {save_dir}")
        print(f"  clusters={encoder.n_clusters}, token_dim={encoder.token_dim}, "
              f"vlad_dim={encoder.vlad_dim}")
        return encoder


class VLADPlaceRecognition:
    """
    基于 VLAD 的场景检索系统。

    工作流:
    1. 从训练集 DINO patch tokens 学习 VLAD 编码器
    2. 对训练集所有帧计算 VLAD 描述子
    3. 构建 FAISS 索引
    4. 对查询图像的 patch tokens 编码 VLAD → 检索 Top-K
    """

    def __init__(self, vlad_encoder: VLADEncoder):
        self.vlad_encoder = vlad_encoder
        self.index: Optional[faiss.IndexFlatIP] = None
        self.frame_ids: List[int] = []
        self.poses_w2c: Optional[np.ndarray] = None
        self.descriptors: Optional[np.ndarray] = None

    @classmethod
    def build_from_features(
        cls,
        feature_dir: str,
        traj_path: str,
        n_clusters: int = 32,
        max_kmeans_samples: int = 100000,
        verbose: bool = True,
    ) -> 'VLADPlaceRecognition':
        """
        从 DINO patch 特征目录构建 VLAD 检索系统。

        Args:
            feature_dir: 包含 fine_dino/ 子目录的特征根目录
            traj_path: C2W 位姿文件
            n_clusters: VLAD 聚类数
            max_kmeans_samples: K-means 最大采样数
        """
        feature_dir = Path(feature_dir)
        dino_dir = feature_dir / 'fine_dino'

        if not dino_dir.exists():
            raise FileNotFoundError(f"DINO 特征目录不存在: {dino_dir}")

        # ── 加载位姿 ──
        traj = np.loadtxt(traj_path)
        c2w_poses = traj.reshape(-1, 4, 4).astype(np.float32)
        all_w2c = np.linalg.inv(c2w_poses).astype(np.float32)

        # ── 扫描帧 ──
        file_map: Dict[int, Path] = {}
        for fpath in dino_dir.glob('rgb_*_fine_dino_*.pt'):
            match = re.search(r'rgb_(\d+)_fine_dino_', fpath.name)
            if match:
                fid = int(match.group(1))
                if fid < len(all_w2c):
                    file_map[fid] = fpath

        sorted_fids = sorted(file_map.keys())
        n_frames = len(sorted_fids)

        if verbose:
            print(f"[VLADPlaceRecognition] {n_frames} frames from {dino_dir}")

        # ── Step 1: 收集 patch tokens 用于 K-means ──
        if verbose:
            print("  Step 1: 加载 patch tokens for K-means...")

        all_patches = []
        patch_tokens_per_frame = []  # 保存用于编码
        first_shape = None

        for i, fid in enumerate(sorted_fids):
            feat = torch.load(str(file_map[fid]), map_location='cpu').numpy()
            # feat: [768, H, W] → patches: [H*W, 768]
            D, H, W = feat.shape
            if first_shape is None:
                first_shape = (D, H, W)
            patches = feat.reshape(D, -1).T  # [H*W, D]
            patch_tokens_per_frame.append(patches)
            all_patches.append(patches)

        all_patches = np.concatenate(all_patches, axis=0)  # [N_total, 768]
        if verbose:
            print(f"  总 patch tokens: {all_patches.shape[0]:,} × {all_patches.shape[1]}d")

        # ── Step 2: 学习 VLAD 编码器 ──
        vlad_encoder = VLADEncoder(n_clusters=n_clusters, token_dim=first_shape[0])
        vlad_encoder.fit(all_patches, max_samples=max_kmeans_samples, verbose=verbose)

        # ── Step 3: 编码所有帧 ──
        if verbose:
            print("  Step 3: 编码 VLAD 描述子...")

        descriptors = vlad_encoder.encode_batch(patch_tokens_per_frame)
        # descriptors: [N, K*D]

        if verbose:
            print(f"  VLAD 描述子: {descriptors.shape}")

        # ── Step 4: 构建 FAISS 索引 ──
        vlad_dim = descriptors.shape[1]
        faiss.normalize_L2(descriptors)  # 确保 L2 归一化

        index = faiss.IndexFlatIP(vlad_dim)
        index.add(descriptors)

        # ── 组装 ──
        system = cls(vlad_encoder=vlad_encoder)
        system.index = index
        system.frame_ids = sorted_fids
        system.poses_w2c = np.stack([all_w2c[fid] for fid in sorted_fids])
        system.descriptors = descriptors

        if verbose:
            print(f"  FAISS 索引: {index.ntotal} vectors, {vlad_dim}d")
            print(f"  (CLS token: 768d → VLAD: {vlad_dim}d, 信息量 ×{vlad_dim/768:.0f})")

        return system

    def query(
        self,
        patch_tokens: Union[torch.Tensor, np.ndarray],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        检索与查询帧最相似的 Top-K 训练帧。

        Args:
            patch_tokens: [n_patches, D] 或 [D, H, W] 查询帧的 DINO patch tokens
            top_k: 返回候选刷

        Returns:
            列表，每项: {frame_id, score, pose_w2c, rank}
        """
        if isinstance(patch_tokens, torch.Tensor):
            patch_tokens = patch_tokens.detach().cpu().numpy()

        # 如果输入是 [D, H, W] 格式，转换为 [H*W, D]
        if patch_tokens.ndim == 3:
            D, H, W = patch_tokens.shape
            patch_tokens = patch_tokens.reshape(D, -1).T

        patch_tokens = patch_tokens.astype(np.float32)

        # 编码 VLAD
        vlad = self.vlad_encoder.encode_single(patch_tokens).reshape(1, -1)
        faiss.normalize_L2(vlad)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vlad, top_k)

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

    def query_from_cls(self, cls_token, **kwargs):
        """不支持 CLS token 查询 (VLAD 需要 patch tokens)。"""
        raise NotImplementedError(
            "VLAD 检索需要 patch tokens, 不支持 CLS token。"
            "请使用 query(patch_tokens) 或 PlaceRecognition 类。"
        )

    def save(self, save_dir: str):
        """保存完整检索系统。"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存 VLAD 编码器
        self.vlad_encoder.save(str(save_dir / 'vlad_encoder'))

        # 保存 FAISS 索引
        faiss.write_index(self.index, str(save_dir / 'vlad_index.faiss'))

        # 保存元数据
        np.save(save_dir / 'frame_ids.npy', np.array(self.frame_ids))
        np.save(save_dir / 'poses_w2c.npy', self.poses_w2c)
        np.save(save_dir / 'descriptors.npy', self.descriptors)

        print(f"[VLADPlaceRecognition] 保存: {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> 'VLADPlaceRecognition':
        """加载已保存的检索系统。"""
        save_dir = Path(save_dir)

        vlad_encoder = VLADEncoder.load(str(save_dir / 'vlad_encoder'))
        system = cls(vlad_encoder=vlad_encoder)

        system.index = faiss.read_index(str(save_dir / 'vlad_index.faiss'))
        system.frame_ids = np.load(save_dir / 'frame_ids.npy').tolist()
        system.poses_w2c = np.load(save_dir / 'poses_w2c.npy')
        system.descriptors = np.load(save_dir / 'descriptors.npy')

        print(f"[VLADPlaceRecognition] 加载: {save_dir}")
        print(f"  FAISS: {system.index.ntotal} vectors, {system.vlad_encoder.vlad_dim}d")
        return system
