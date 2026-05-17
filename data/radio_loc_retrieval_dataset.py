"""
Dataset and utilities for concat-localizer evaluation under real pose init.

This keeps the training path untouched and swaps only the evaluation-time
initial pose source:
  - retrieval_cls: DINO CLS-token cosine retrieval against the train split
  - fallback_nearest_train_pose_gt: GT-assisted nearest train pose fallback
  - fallback_synthetic_noise: synthetic noise fallback
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    faiss = None

from data.radio_loc_dataset import (
    RadioLocDataset,
    add_pose_noise,
    colmap_to_w2c,
    read_colmap_images,
)

OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS = (
    "retrieval_original_scores_candidates",
    "retrieval_pnp_success_candidates",
    "retrieval_pnp_num_inliers_candidates",
    "retrieval_pnp_num_matches_candidates",
    "retrieval_pnp_reproj_rmse_candidates",
    "retrieval_pnp_reproj_median_candidates",
    "retrieval_pnp_inlier_ratio_candidates",
    "retrieval_pnp_inlier_conf_mean_candidates",
)


def _safe_torch_load(path: str) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_image_stem(image_name: str) -> str:
    return os.path.splitext(image_name.replace("\\", "/"))[0].replace("/", "_")


def parse_split_names(split_file: Optional[str]) -> Optional[set]:
    if split_file is None or not os.path.isfile(split_file):
        return None
    split_names = set()
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Visual") or line.startswith("ImageFile"):
                continue
            image_name = line.split()[0]
            base = os.path.splitext(image_name)[0]
            split_names.add(base + ".png")
            split_names.add(base + ".jpg")
    return split_names


def list_colmap_split_samples(colmap_dir: str, split_file: Optional[str]) -> List[Dict]:
    images = read_colmap_images(os.path.join(colmap_dir, "images.bin"))
    split_names = parse_split_names(split_file)

    samples: List[Dict] = []
    for img_id in sorted(images.keys()):
        meta = images[img_id]
        if split_names is not None and meta.name not in split_names:
            continue
        samples.append(
            {
                "img_id": img_id,
                "image_name": meta.name,
                "image_stem": normalize_image_stem(meta.name),
                "pose_w2c": colmap_to_w2c(meta.qvec, meta.tvec).astype(np.float32),
            }
        )
    return samples


def _camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    return (-R.T @ t).astype(np.float32)


def _find_cls_feature_path(retrieval_feature_dir: str, image_stem: str) -> Optional[str]:
    cls_dir = Path(retrieval_feature_dir) / "cls"
    if not cls_dir.is_dir():
        return None
    matches = sorted(cls_dir.glob(f"{image_stem}_cls_*.pt"))
    return str(matches[0]) if matches else None


def _load_cls_descriptor(path: str) -> np.ndarray:
    token = _safe_torch_load(path).float().view(-1).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(token)
    return token / max(norm, 1e-8)


def _search_cosine_topk(query: np.ndarray, database: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    topk = int(max(1, min(topk, len(database))))
    if faiss is not None:
        index = faiss.IndexFlatIP(database.shape[1])
        db = database.astype(np.float32).copy()
        faiss.normalize_L2(db)
        index.add(db)
        q = query.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(q)
        scores, indices = index.search(q, topk)
        return indices[0].astype(np.int64), scores[0].astype(np.float32)

    sims = database @ query.reshape(-1, 1)
    order = np.argsort(-sims[:, 0])[:topk]
    return order.astype(np.int64), sims[order, 0].astype(np.float32)


def _search_cosine_top1(query: np.ndarray, database: np.ndarray) -> Tuple[int, float]:
    indices, scores = _search_cosine_topk(query, database, topk=1)
    return int(indices[0]), float(scores[0])


def _resolve_method(method: str, retrieval_feature_dir: Optional[str]) -> str:
    if method != "auto":
        return method
    if retrieval_feature_dir and os.path.isdir(os.path.join(retrieval_feature_dir, "cls")):
        return "cls"
    return "nearest_train_pose_gt"


def build_retrieval_init_entries(
    colmap_dir: str,
    train_split_file: str,
    query_split_file: str,
    retrieval_feature_dir: Optional[str] = None,
    method: str = "auto",
    topk: int = 1,
    exclude_query_from_db: bool = False,
    fallback_mode: str = "nearest_train_pose_gt",
    fallback_noise_rot_deg: float = 3.0,
    fallback_noise_trans_m: float = 0.10,
) -> Tuple[List[Dict], Dict]:
    train_samples = list_colmap_split_samples(colmap_dir, train_split_file)
    query_samples = list_colmap_split_samples(colmap_dir, query_split_file)
    if not train_samples:
        raise RuntimeError(f"No train samples found for retrieval: {train_split_file}")
    if not query_samples:
        raise RuntimeError(f"No query samples found for retrieval: {query_split_file}")

    effective_method = _resolve_method(method, retrieval_feature_dir)
    db_descriptors: List[np.ndarray] = []
    db_samples: List[Dict] = []

    if effective_method == "cls":
        if not retrieval_feature_dir:
            raise ValueError("retrieval_feature_dir is required for cls retrieval")
        for sample in train_samples:
            feature_path = _find_cls_feature_path(retrieval_feature_dir, sample["image_stem"])
            if feature_path is None:
                continue
            db_samples.append(sample)
            db_descriptors.append(_load_cls_descriptor(feature_path))
        if not db_descriptors:
            effective_method = fallback_mode

    db_matrix = (
        np.stack(db_descriptors, axis=0).astype(np.float32)
        if db_descriptors
        else np.zeros((0, 1), dtype=np.float32)
    )

    train_centers = np.stack(
        [_camera_center_from_w2c(sample["pose_w2c"]) for sample in train_samples], axis=0
    )

    entries: List[Dict] = []
    stats = {
        "method_requested": method,
        "method_used": effective_method,
        "retrieval_topk_requested": int(max(1, topk)),
        "exclude_query_from_db": bool(exclude_query_from_db),
        "num_train_samples": len(train_samples),
        "num_query_samples": len(query_samples),
        "num_retrieval_db": len(db_samples),
        "counts_by_source": {},
    }

    counts_by_source: Dict[str, int] = {}
    for query in query_samples:
        source = ""
        pose_init = None
        retrieval_frame_id = -1
        retrieval_image_name = ""
        retrieval_score = float("nan")
        candidate_poses = None
        candidate_frame_ids = None
        candidate_image_names = None
        candidate_scores = None
        candidate_valid_mask = None

        if effective_method == "cls":
            query_path = _find_cls_feature_path(retrieval_feature_dir, query["image_stem"])
            if query_path is not None and len(db_samples) > 0:
                query_desc = _load_cls_descriptor(query_path)
                if exclude_query_from_db:
                    sims = (db_matrix @ query_desc.reshape(-1, 1))[:, 0]
                    valid_indices = np.array(
                        [
                            idx
                            for idx, sample in enumerate(db_samples)
                            if sample["image_name"] != query["image_name"]
                        ],
                        dtype=np.int64,
                    )
                    if len(valid_indices) > 0:
                        ordered = valid_indices[np.argsort(-sims[valid_indices])]
                        candidate_indices = ordered[: max(1, topk)]
                        candidate_scores_np = sims[candidate_indices].astype(np.float32)
                    else:
                        candidate_indices = np.zeros((0,), dtype=np.int64)
                        candidate_scores_np = np.zeros((0,), dtype=np.float32)
                else:
                    candidate_indices, candidate_scores_np = _search_cosine_topk(
                        query_desc, db_matrix, topk=max(1, topk)
                    )
                if len(candidate_indices) == 0:
                    candidate_indices = np.zeros((0,), dtype=np.int64)
                    candidate_scores_np = np.zeros((0,), dtype=np.float32)
                selected_samples = [db_samples[int(idx)] for idx in candidate_indices]
                selected_poses = [sample["pose_w2c"].copy() for sample in selected_samples]
                valid_count = len(selected_poses)
                pad_count = max(1, topk) - valid_count
                if valid_count > 0 and pad_count > 0:
                    selected_samples.extend([selected_samples[0]] * pad_count)
                    selected_poses.extend([selected_poses[0].copy()] * pad_count)
                    candidate_scores_np = np.concatenate(
                        [candidate_scores_np, np.repeat(candidate_scores_np[:1], pad_count)]
                    )
                if valid_count > 0:
                    candidate_poses = np.stack(selected_poses, axis=0).astype(np.float32)
                    candidate_frame_ids = np.array(
                        [int(sample["img_id"]) for sample in selected_samples], dtype=np.int64
                    )
                    candidate_image_names = np.array(
                        [sample["image_name"] for sample in selected_samples]
                    )
                    candidate_scores = candidate_scores_np.astype(np.float32)
                    candidate_valid_mask = np.zeros((max(1, topk),), dtype=bool)
                    candidate_valid_mask[:valid_count] = True

                    db_sample = selected_samples[0]
                    pose_init = candidate_poses[0].copy()
                    retrieval_frame_id = int(db_sample["img_id"])
                    retrieval_image_name = db_sample["image_name"]
                    retrieval_score = float(candidate_scores[0])
                    source = "retrieval_cls"

        if pose_init is None:
            if fallback_mode == "synthetic_noise":
                pose_init = add_pose_noise(
                    query["pose_w2c"], fallback_noise_rot_deg, fallback_noise_trans_m
                ).astype(np.float32)
                source = "fallback_synthetic_noise"
            else:
                query_center = _camera_center_from_w2c(query["pose_w2c"])
                nearest_idx = int(np.argmin(np.linalg.norm(train_centers - query_center[None], axis=1)))
                db_sample = train_samples[nearest_idx]
                pose_init = db_sample["pose_w2c"].copy()
                retrieval_frame_id = int(db_sample["img_id"])
                retrieval_image_name = db_sample["image_name"]
                retrieval_score = float(
                    np.linalg.norm(train_centers[nearest_idx] - query_center)
                )
                source = "fallback_nearest_train_pose_gt"

            candidate_poses = np.repeat(pose_init.astype(np.float32)[None], max(1, topk), axis=0)
            candidate_frame_ids = np.full((max(1, topk),), retrieval_frame_id, dtype=np.int64)
            candidate_image_names = np.array([retrieval_image_name] * max(1, topk))
            candidate_scores = np.full((max(1, topk),), retrieval_score, dtype=np.float32)
            candidate_valid_mask = np.zeros((max(1, topk),), dtype=bool)
            candidate_valid_mask[0] = True

        counts_by_source[source] = counts_by_source.get(source, 0) + 1
        entries.append(
            {
                "query_img_id": int(query["img_id"]),
                "query_image_name": query["image_name"],
                "query_image_stem": query["image_stem"],
                "pose_init": pose_init.astype(np.float32),
                "init_source": source,
                "retrieval_frame_id": retrieval_frame_id,
                "retrieval_image_name": retrieval_image_name,
                "retrieval_score": retrieval_score,
                "pose_init_candidates": candidate_poses.astype(np.float32),
                "candidate_valid_mask": candidate_valid_mask.astype(bool),
                "retrieval_frame_ids_candidates": candidate_frame_ids.astype(np.int64),
                "retrieval_image_names_candidates": candidate_image_names,
                "retrieval_scores_candidates": candidate_scores.astype(np.float32),
            }
        )

    stats["counts_by_source"] = counts_by_source
    return entries, stats


def save_retrieval_init_entries(entries: Sequence[Dict], stats: Dict, save_path: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query_img_ids": np.array([e["query_img_id"] for e in entries], dtype=np.int64),
        "query_image_names": np.array([e["query_image_name"] for e in entries]),
        "query_image_stems": np.array([e["query_image_stem"] for e in entries]),
        "pose_inits": np.stack([e["pose_init"] for e in entries]).astype(np.float32),
        "init_sources": np.array([e["init_source"] for e in entries]),
        "retrieval_frame_ids": np.array([e["retrieval_frame_id"] for e in entries], dtype=np.int64),
        "retrieval_image_names": np.array([e["retrieval_image_name"] for e in entries]),
        "retrieval_scores": np.array([e["retrieval_score"] for e in entries], dtype=np.float32),
        "pose_init_candidates": np.stack([e["pose_init_candidates"] for e in entries]).astype(np.float32),
        "candidate_valid_mask": np.stack([e["candidate_valid_mask"] for e in entries]).astype(bool),
        "retrieval_frame_ids_candidates": np.stack(
            [e["retrieval_frame_ids_candidates"] for e in entries]
        ).astype(np.int64),
        "retrieval_image_names_candidates": np.stack(
            [e["retrieval_image_names_candidates"] for e in entries]
        ),
        "retrieval_scores_candidates": np.stack(
            [e["retrieval_scores_candidates"] for e in entries]
        ).astype(np.float32),
        "stats": np.array([stats], dtype=object),
    }
    for key in OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS:
        if all(key in e for e in entries):
            payload[key] = np.stack([e[key] for e in entries]).astype(np.float32)
    np.savez(save_path, **payload)


def load_retrieval_init_entries(load_path: str) -> Tuple[List[Dict], Dict]:
    data = np.load(load_path, allow_pickle=True)
    files = set(data.files)
    if "query_img_ids" in files:
        num_entries = len(data["query_img_ids"])
        query_img_ids = data["query_img_ids"]
    else:
        num_entries = len(data["pose_inits"])
        query_img_ids = np.arange(num_entries, dtype=np.int64)
    query_image_names = data["query_image_names"]
    query_image_stems = (
        data["query_image_stems"]
        if "query_image_stems" in files
        else np.array([Path(str(name)).stem for name in query_image_names])
    )
    init_sources = (
        data["init_sources"]
        if "init_sources" in files
        else np.array(["pose_init_cache"] * num_entries)
    )
    retrieval_frame_ids = (
        data["retrieval_frame_ids"]
        if "retrieval_frame_ids" in files
        else np.full((num_entries,), -1, dtype=np.int64)
    )
    retrieval_image_names = (
        data["retrieval_image_names"]
        if "retrieval_image_names" in files
        else query_image_names
    )
    retrieval_scores = (
        data["retrieval_scores"]
        if "retrieval_scores" in files
        else np.zeros((num_entries,), dtype=np.float32)
    )
    entries: List[Dict] = []
    for idx in range(num_entries):
        entries.append(
            {
                "query_img_id": int(query_img_ids[idx]),
                "query_image_name": str(query_image_names[idx]),
                "query_image_stem": str(query_image_stems[idx]),
                "pose_init": data["pose_inits"][idx].astype(np.float32),
                "init_source": str(init_sources[idx]),
                "retrieval_frame_id": int(retrieval_frame_ids[idx]),
                "retrieval_image_name": str(retrieval_image_names[idx]),
                "retrieval_score": float(retrieval_scores[idx]),
            }
        )
        if "pose_init_candidates" in data.files:
            entries[-1]["pose_init_candidates"] = data["pose_init_candidates"][idx].astype(np.float32)
            entries[-1]["candidate_valid_mask"] = data["candidate_valid_mask"][idx].astype(bool)
            entries[-1]["retrieval_frame_ids_candidates"] = data[
                "retrieval_frame_ids_candidates"
            ][idx].astype(np.int64)
            entries[-1]["retrieval_image_names_candidates"] = [
                str(v) for v in data["retrieval_image_names_candidates"][idx]
            ]
            entries[-1]["retrieval_scores_candidates"] = data[
                "retrieval_scores_candidates"
            ][idx].astype(np.float32)
            for key in OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS:
                if key in data.files:
                    entries[-1][key] = data[key][idx].astype(np.float32)
        else:
            entries[-1]["pose_init_candidates"] = entries[-1]["pose_init"][None].astype(np.float32)
            entries[-1]["candidate_valid_mask"] = np.array([True], dtype=bool)
            entries[-1]["retrieval_frame_ids_candidates"] = np.array(
                [entries[-1]["retrieval_frame_id"]], dtype=np.int64
            )
            entries[-1]["retrieval_image_names_candidates"] = [entries[-1]["retrieval_image_name"]]
            entries[-1]["retrieval_scores_candidates"] = np.array(
                [entries[-1]["retrieval_score"]], dtype=np.float32
            )
    if "stats" in data.files:
        stats_obj = data["stats"][0]
        stats = stats_obj.item() if hasattr(stats_obj, "item") else stats_obj
    else:
        stats = {}
    return entries, stats


class RadioLocRetrievalDataset(RadioLocDataset):
    """Evaluation dataset with retrieval-based or explicit init poses."""

    def __init__(
        self,
        *args,
        source_dir: Optional[str] = None,
        init_poses_path: Optional[str] = None,
        retrieval_feature_dir: Optional[str] = None,
        retrieval_train_split: Optional[str] = None,
        retrieval_method: str = "auto",
        retrieval_topk: int = 1,
        exclude_query_from_db: bool = False,
        fallback_mode: str = "nearest_train_pose_gt",
        fallback_noise_rot_deg: float = 3.0,
        fallback_noise_trans_m: float = 0.10,
        save_init_poses_path: Optional[str] = None,
        jitter_loaded_init: bool = False,
        sample_topk_init: bool = False,
        sample_topk_prob: float = 0.0,
        **kwargs,
    ):
        self.colmap_dir = kwargs["colmap_dir"]
        self.split_file = kwargs.get("split_file")
        self.source_dir = source_dir
        self.jitter_loaded_init = bool(jitter_loaded_init)
        self.sample_topk_init = bool(sample_topk_init)
        self.sample_topk_prob = float(sample_topk_prob)
        self.init_stats: Dict = {}
        super().__init__(*args, source_dir=source_dir, **kwargs)

        image_meta = {s["img_id"]: s for s in list_colmap_split_samples(self.colmap_dir, None)}
        for sample in self.samples:
            meta = image_meta[sample["img_id"]]
            sample["image_name"] = meta["image_name"]
            sample["image_stem"] = meta["image_stem"]

        entries: List[Dict]
        if init_poses_path and os.path.isfile(init_poses_path):
            entries, self.init_stats = load_retrieval_init_entries(init_poses_path)
        else:
            entries, self.init_stats = build_retrieval_init_entries(
                colmap_dir=self.colmap_dir,
                train_split_file=retrieval_train_split,
                query_split_file=self.split_file,
                retrieval_feature_dir=retrieval_feature_dir,
                method=retrieval_method,
                topk=retrieval_topk,
                exclude_query_from_db=exclude_query_from_db,
                fallback_mode=fallback_mode,
                fallback_noise_rot_deg=fallback_noise_rot_deg,
                fallback_noise_trans_m=fallback_noise_trans_m,
            )
            if save_init_poses_path:
                save_retrieval_init_entries(entries, self.init_stats, save_init_poses_path)

        self.init_by_name = {entry["query_image_name"]: entry for entry in entries}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        sample = self.samples[idx]
        image_name = sample["image_name"]
        image_stem = sample["image_stem"]
        is_train = getattr(self, "split", None) == "train"
        entry = self.init_by_name.get(image_name)
        if entry is None:
            raise KeyError(f"Missing real-init entry for {image_name}")

        pose_init = entry["pose_init"].copy()
        if is_train and self.sample_topk_init and self.sample_topk_prob > 0.0:
            if np.random.rand() < self.sample_topk_prob:
                valid_mask = np.asarray(entry["candidate_valid_mask"], dtype=bool)
                valid_indices = np.flatnonzero(valid_mask)
                if len(valid_indices) > 0:
                    chosen_idx = int(np.random.choice(valid_indices))
                    pose_init = entry["pose_init_candidates"][chosen_idx].copy()
        if is_train and self.jitter_loaded_init:
            pose_init = add_pose_noise(pose_init, self.noise_rot_deg, self.noise_trans_m).astype(np.float32)

        item["pose_init"] = torch.tensor(np.asarray(pose_init, dtype=np.float32).tolist(), dtype=torch.float32)
        item["image_name"] = image_name
        item["image_stem"] = image_stem
        item["init_source"] = entry["init_source"]
        item["retrieval_frame_id"] = entry["retrieval_frame_id"]
        item["retrieval_image_name"] = entry["retrieval_image_name"]
        item["retrieval_score"] = float(entry["retrieval_score"])
        item["pose_init_candidates"] = torch.tensor(
            entry["pose_init_candidates"].tolist(), dtype=torch.float32
        )
        item["candidate_valid_mask"] = torch.tensor(
            entry["candidate_valid_mask"].tolist(), dtype=torch.bool
        )
        item["retrieval_frame_ids_candidates"] = torch.tensor(
            entry["retrieval_frame_ids_candidates"].tolist(), dtype=torch.long
        )
        item["retrieval_scores_candidates"] = torch.tensor(
            entry["retrieval_scores_candidates"].tolist(), dtype=torch.float32
        )
        for key in OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS:
            if key in entry:
                item[key] = torch.tensor(entry[key].tolist(), dtype=torch.float32)
        item["retrieval_image_names_candidates"] = list(entry["retrieval_image_names_candidates"])

        query_rgb_path = ""
        if self.source_dir:
            candidate = os.path.join(self.source_dir, image_name)
            if os.path.isfile(candidate):
                query_rgb_path = candidate
        item["query_rgb_path"] = query_rgb_path
        return item
