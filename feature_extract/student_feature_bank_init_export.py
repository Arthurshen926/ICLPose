#!/usr/bin/env python3
"""Export student-feature-bank retrieval poses as real-init caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_retrieval_dataset import (  # noqa: E402
    list_colmap_split_samples,
    save_retrieval_init_entries,
)
from feature_extract.pose_init_export import _load_rgb, _records_by_sample_name  # noqa: E402
from feature_extract.train_impl import (  # noqa: E402
    TeacherFeatureStore,
    build_all_records,
    build_radio_query_student,
    load_config,
    resolve_query_feature_dims,
    safe_torch_load,
)


def _load_feature_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True).float()
    except TypeError:
        return torch.load(path, map_location="cpu").float()


def _spatial_mean_feature(feature: torch.Tensor) -> torch.Tensor:
    feat = torch.as_tensor(feature, dtype=torch.float32)
    if feat.ndim == 3:
        feat = feat.unsqueeze(0)
    if feat.ndim == 4:
        return feat.flatten(2).mean(dim=2)
    if feat.ndim == 2:
        return feat
    raise ValueError(f"student feature must have shape [B,C,H,W], [C,H,W], or [B,C], got {tuple(feat.shape)}")


def pool_student_descriptor(
    outputs: Dict[str, torch.Tensor],
    *,
    fine_key: str = "fine",
    coarse_key: str = "coarse",
) -> torch.Tensor:
    """Pool fine/coarse student maps into one L2-normalized image descriptor."""
    if fine_key not in outputs:
        raise KeyError(f"Missing fine student output key: {fine_key}")
    if coarse_key not in outputs:
        raise KeyError(f"Missing coarse student output key: {coarse_key}")
    fine = _spatial_mean_feature(outputs[fine_key])
    coarse = _spatial_mean_feature(outputs[coarse_key])
    if fine.shape[0] != coarse.shape[0]:
        raise ValueError(f"fine/coarse batch sizes differ: {fine.shape[0]} vs {coarse.shape[0]}")
    return F.normalize(torch.cat([fine, coarse], dim=1).float(), dim=1, eps=1e-8)


def extract_cached_student_descriptors(
    feature_dir: str,
    samples: Sequence[Dict],
    *,
    fine_subdir: str = "fine_geo",
    coarse_subdir: str = "coarse_sem",
) -> tuple[list[Dict], torch.Tensor]:
    """Load exported student feature maps by COLMAP image id and pool descriptors."""
    store = TeacherFeatureStore(feature_dir, cache_in_memory=False)
    if fine_subdir != "fine_geo" or coarse_subdir != "coarse_sem":
        # Keep extension point explicit; current cache naming/discovery is tied to these dirs.
        raise ValueError("Only fine_geo/coarse_sem cached feature dirs are currently supported")
    descriptors = []
    used_samples: list[Dict] = []
    for sample in samples:
        img_id = int(sample["img_id"])
        if img_id not in store.fine_files or img_id not in store.coarse_files:
            continue
        outputs = {
            "fine": _load_feature_tensor(store.fine_files[img_id]).unsqueeze(0),
            "coarse": _load_feature_tensor(store.coarse_files[img_id]).unsqueeze(0),
        }
        descriptors.append(pool_student_descriptor(outputs).cpu())
        used_samples.append(sample)
    if not descriptors:
        raise RuntimeError(f"No cached student descriptors were extracted from {feature_dir}")
    return used_samples, torch.cat(descriptors, dim=0)


def search_student_feature_bank(
    query_descriptors: torch.Tensor,
    bank_descriptors: torch.Tensor,
    *,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top-K cosine matches from a normalized or unnormalized descriptor bank."""
    query = F.normalize(torch.as_tensor(query_descriptors, dtype=torch.float32), dim=1, eps=1e-8)
    bank = F.normalize(torch.as_tensor(bank_descriptors, dtype=torch.float32), dim=1, eps=1e-8)
    if query.ndim != 2 or bank.ndim != 2:
        raise ValueError(f"query/bank descriptors must be 2D, got {tuple(query.shape)} and {tuple(bank.shape)}")
    if query.shape[1] != bank.shape[1]:
        raise ValueError(f"descriptor dims differ: query={query.shape[1]} bank={bank.shape[1]}")
    k = max(1, min(int(topk), int(bank.shape[0])))
    scores = query @ bank.t()
    top_scores, top_indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
    return top_indices, top_scores


def build_student_feature_bank_entries(
    *,
    query_samples: Sequence[Dict],
    train_samples: Sequence[Dict],
    indices,
    scores,
    source_name: str,
    save_path: str | None = None,
) -> Tuple[list[Dict], Dict]:
    indices_np = np.asarray(torch.as_tensor(indices, dtype=torch.long).cpu().numpy())
    scores_np = np.asarray(torch.as_tensor(scores, dtype=torch.float32).cpu().numpy(), dtype=np.float32)
    if indices_np.ndim != 2 or scores_np.ndim != 2:
        raise ValueError(f"indices/scores must have shape [N,K], got {indices_np.shape} and {scores_np.shape}")
    if indices_np.shape != scores_np.shape:
        raise ValueError(f"indices and scores shapes differ: {indices_np.shape} vs {scores_np.shape}")
    if len(query_samples) != indices_np.shape[0]:
        raise ValueError(f"query_samples length {len(query_samples)} does not match indices {indices_np.shape[0]}")
    if len(train_samples) == 0:
        raise ValueError("train_samples is empty")

    topk = indices_np.shape[1]
    entries = []
    for qidx, sample in enumerate(query_samples):
        order = np.argsort(-scores_np[qidx])
        matched_indices = indices_np[qidx, order]
        matched_scores = scores_np[qidx, order]

        candidate_poses = []
        candidate_frame_ids = []
        candidate_image_names = []
        for bank_idx in matched_indices:
            train_sample = train_samples[int(bank_idx)]
            candidate_poses.append(np.asarray(train_sample["pose_w2c"], dtype=np.float32))
            candidate_frame_ids.append(int(train_sample["img_id"]))
            candidate_image_names.append(str(train_sample["image_name"]))

        entries.append(
            {
                "query_img_id": int(sample["img_id"]),
                "query_image_name": str(sample["image_name"]),
                "query_image_stem": str(sample["image_stem"]),
                "pose_init": candidate_poses[0].astype(np.float32),
                "init_source": source_name,
                "retrieval_frame_id": int(candidate_frame_ids[0]),
                "retrieval_image_name": str(candidate_image_names[0]),
                "retrieval_score": float(matched_scores[0]),
                "pose_init_candidates": np.stack(candidate_poses, axis=0).astype(np.float32),
                "candidate_valid_mask": np.ones((topk,), dtype=bool),
                "retrieval_frame_ids_candidates": np.asarray(candidate_frame_ids, dtype=np.int64),
                "retrieval_image_names_candidates": np.asarray(candidate_image_names),
                "retrieval_scores_candidates": matched_scores.astype(np.float32),
            }
        )

    stats = {
        "method_requested": "student_feature_bank",
        "method_used": source_name,
        "retrieval_topk_requested": int(topk),
        "num_query_samples": int(len(entries)),
        "num_train_samples": int(len(train_samples)),
        "counts_by_source": {source_name: int(len(entries))},
    }
    if save_path:
        save_retrieval_init_entries(entries, stats, save_path)
    return entries, stats


def _match_samples_to_records(samples: Sequence[Dict], record_by_name: Dict[str, Dict]) -> list[tuple[Dict, Dict]]:
    matched = []
    for sample in samples:
        record = record_by_name.get(sample["image_name"]) or record_by_name.get(Path(sample["image_name"]).name)
        if record is None or record.get("image_path") is None:
            continue
        matched.append((sample, record))
    return matched


@torch.no_grad()
def _extract_student_descriptors(
    *,
    model,
    sample_record_pairs: Sequence[tuple[Dict, Dict]],
    input_hw: tuple[int, int],
    batch_size: int,
    device: torch.device,
    fine_key: str,
    coarse_key: str,
) -> tuple[list[Dict], torch.Tensor]:
    descriptors = []
    used_samples: list[Dict] = []
    for start in range(0, len(sample_record_pairs), int(batch_size)):
        chunk = sample_record_pairs[start : start + int(batch_size)]
        rgb_batch = torch.stack([_load_rgb(record["image_path"], input_hw) for _sample, record in chunk], dim=0).to(device)
        outputs = model(rgb_batch)
        descriptors.append(pool_student_descriptor(outputs, fine_key=fine_key, coarse_key=coarse_key).cpu())
        used_samples.extend([sample for sample, _record in chunk])
    if not descriptors:
        raise RuntimeError("No student descriptors were extracted; check split/image paths")
    return used_samples, torch.cat(descriptors, dim=0)


@torch.no_grad()
def export_student_feature_bank_init(
    *,
    config_path: str,
    checkpoint_path: str,
    cached_feature_dir: str | None = None,
    colmap_dir: str,
    train_split: str,
    query_split: str,
    save_path: str,
    batch_size: int = 4,
    device: str = "cuda",
    topk: int = 10,
    source_name: str | None = None,
    fine_key: str = "fine",
    coarse_key: str = "coarse",
) -> Dict:
    train_samples = list_colmap_split_samples(colmap_dir, train_split)
    query_samples = list_colmap_split_samples(colmap_dir, query_split)
    descriptor_source = "cached_student_features" if cached_feature_dir else "online_student_model"
    if cached_feature_dir:
        used_train_samples, bank_desc = extract_cached_student_descriptors(cached_feature_dir, train_samples)
        used_query_samples, query_desc = extract_cached_student_descriptors(cached_feature_dir, query_samples)
    else:
        cfg = load_config(config_path)
        device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        teacher_store = TeacherFeatureStore(
            cfg["dataset"]["feature_dir"],
            cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
        )
        fine_dim, coarse_dim = resolve_query_feature_dims(cfg, teacher_store)
        model = build_radio_query_student(cfg, fine_feature_dim=fine_dim, coarse_feature_dim=coarse_dim).to(device_obj)
        checkpoint = safe_torch_load(checkpoint_path)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        model.eval()

        all_records = build_all_records(cfg["dataset"], teacher_store, allow_synthetic=False)
        record_by_name = _records_by_sample_name(all_records)
        train_pairs = _match_samples_to_records(train_samples, record_by_name)
        query_pairs = _match_samples_to_records(query_samples, record_by_name)
        if not train_pairs:
            raise RuntimeError("No train split samples matched RGB records")
        if not query_pairs:
            raise RuntimeError("No query split samples matched RGB records")

        input_hw = tuple(cfg["dataset"]["input_hw"])
        used_train_samples, bank_desc = _extract_student_descriptors(
            model=model,
            sample_record_pairs=train_pairs,
            input_hw=input_hw,
            batch_size=batch_size,
            device=device_obj,
            fine_key=fine_key,
            coarse_key=coarse_key,
        )
        used_query_samples, query_desc = _extract_student_descriptors(
            model=model,
            sample_record_pairs=query_pairs,
            input_hw=input_hw,
            batch_size=batch_size,
            device=device_obj,
            fine_key=fine_key,
            coarse_key=coarse_key,
        )
    indices, scores = search_student_feature_bank(query_desc, bank_desc, topk=topk)
    source = source_name or f"student_feature_bank_{Path(checkpoint_path).parent.parent.name}"
    entries, stats = build_student_feature_bank_entries(
        query_samples=used_query_samples,
        train_samples=used_train_samples,
        indices=indices,
        scores=scores,
        source_name=source,
        save_path=None,
    )
    stats.update(
        {
            "descriptor_dim": int(bank_desc.shape[1]),
            "descriptor_source": descriptor_source,
            "num_query_samples_requested": int(len(query_samples)),
            "num_train_samples_requested": int(len(train_samples)),
        }
    )
    save_retrieval_init_entries(entries, stats, save_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export student-feature-bank real-init cache")
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--cached_feature_dir", default=None)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--source_name", default=None)
    parser.add_argument("--fine_key", default="fine")
    parser.add_argument("--coarse_key", default="coarse")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_student_feature_bank_init(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        cached_feature_dir=args.cached_feature_dir,
        colmap_dir=args.colmap_dir,
        train_split=args.train_split,
        query_split=args.query_split,
        save_path=args.save_path,
        batch_size=args.batch_size,
        device=args.device,
        topk=args.topk,
        source_name=args.source_name,
        fine_key=args.fine_key,
        coarse_key=args.coarse_key,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
