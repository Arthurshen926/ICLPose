#!/usr/bin/env python3
"""Export coarse-feature pose-bank retrieval caches.

This is the first step toward an implicit coarse-to-fine localization path:
query/map coarse features produce pose anchors using the same init-cache schema
as the existing evaluation code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_retrieval_dataset import (  # noqa: E402
    list_colmap_split_samples,
    save_retrieval_init_entries,
)
from feature_retrieval.localization_mainline import pose_error  # noqa: E402


def _load_feature_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True).float()
    except TypeError:
        return torch.load(path, map_location="cpu").float()


def _as_bchw(feature: torch.Tensor) -> torch.Tensor:
    feat = torch.as_tensor(feature, dtype=torch.float32)
    if feat.ndim == 2:
        return feat
    if feat.ndim == 3:
        return feat.unsqueeze(0)
    if feat.ndim == 4:
        return feat
    raise ValueError(f"coarse feature must have shape [C,H,W], [B,C,H,W], or [B,C], got {tuple(feat.shape)}")


def _as_b1hw(mask: torch.Tensor, *, batch_size: int, hw: tuple[int, int]) -> torch.Tensor:
    mask_t = torch.as_tensor(mask, dtype=torch.float32)
    if mask_t.ndim == 2:
        mask_t = mask_t.unsqueeze(0).unsqueeze(0)
    elif mask_t.ndim == 3:
        mask_t = mask_t.unsqueeze(1)
    elif mask_t.ndim != 4:
        raise ValueError(f"mask must have shape [H,W], [B,H,W], or [B,1,H,W], got {tuple(mask_t.shape)}")
    if mask_t.shape[0] == 1 and batch_size > 1:
        mask_t = mask_t.expand(batch_size, -1, -1, -1)
    if mask_t.shape[0] != batch_size:
        raise ValueError(f"mask batch size {mask_t.shape[0]} does not match feature batch size {batch_size}")
    if tuple(mask_t.shape[-2:]) != tuple(hw):
        mask_t = F.interpolate(mask_t, size=hw, mode="nearest")
    return mask_t


def pool_coarse_descriptor(feature: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pool coarse feature maps into one L2-normalized descriptor per image."""
    feat = _as_bchw(feature)
    if feat.ndim == 2:
        return F.normalize(feat.float(), dim=1, eps=1e-8)

    if mask is None:
        desc = feat.flatten(2).mean(dim=2)
    else:
        mask_t = _as_b1hw(mask, batch_size=int(feat.shape[0]), hw=(int(feat.shape[-2]), int(feat.shape[-1])))
        denom = torch.clamp(mask_t.flatten(2).sum(dim=2), min=1e-6)
        desc = (feat * mask_t).flatten(2).sum(dim=2) / denom
    return F.normalize(desc.float(), dim=1, eps=1e-8)


def _find_coarse_feature_path(feature_dir: Path, img_id: int) -> Path | None:
    coarse_dir = feature_dir / "coarse_sem"
    if not coarse_dir.is_dir():
        return None
    matches = sorted(coarse_dir.glob(f"rgb_{int(img_id)}_coarse_sem_*.pt"))
    if matches:
        return matches[0]
    fallback = sorted(coarse_dir.glob(f"*_{int(img_id)}_coarse_sem_*.pt"))
    return fallback[0] if fallback else None


def extract_cached_coarse_descriptors(
    feature_dir: str,
    samples: Sequence[Dict],
) -> tuple[list[Dict], torch.Tensor]:
    """Load exported coarse_sem feature maps by COLMAP image id and pool descriptors."""
    root = Path(feature_dir)
    descriptors = []
    used_samples: list[Dict] = []
    for sample in samples:
        path = _find_coarse_feature_path(root, int(sample["img_id"]))
        if path is None:
            continue
        descriptors.append(pool_coarse_descriptor(_load_feature_tensor(path)).cpu())
        used_samples.append(sample)
    if not descriptors:
        raise RuntimeError(f"No cached coarse descriptors were extracted from {feature_dir}")
    return used_samples, torch.cat(descriptors, dim=0)


def extract_rendered_map_coarse_descriptors_from_renderer(
    map_renderer,
    samples: Sequence[Dict],
) -> tuple[list[Dict], torch.Tensor]:
    """Render map-side coarse descriptors at each sample pose using a MapFeatureRenderer-like object."""
    descriptors = []
    used_samples: list[Dict] = []
    for sample in samples:
        try:
            _fine_raw, _fine, coarse, mask, *_rest = map_renderer._render_single(
                str(sample["image_name"]),
                require_grad=False,
            )
        except KeyError:
            continue
        descriptors.append(pool_coarse_descriptor(coarse.detach().cpu(), mask=mask.detach().cpu()).cpu())
        used_samples.append(sample)
    if not descriptors:
        raise RuntimeError("No rendered map coarse descriptors were extracted")
    return used_samples, torch.cat(descriptors, dim=0)


def extract_rendered_map_coarse_descriptors(
    *,
    config_path: str,
    checkpoint_path: str,
    samples: Sequence[Dict],
    device: str = "cuda",
) -> tuple[list[Dict], torch.Tensor]:
    """Build the DCFF map renderer from a feature config/checkpoint and pool map-side coarse descriptors."""
    from feature_extract.train_impl import (  # noqa: WPS433 - keeps CLI import light
        MapFeatureRenderer,
        load_feature_extract_config,
        safe_torch_load,
    )

    class _Logger:
        def info(self, *args, **_kwargs):
            if args:
                try:
                    print(str(args[0]) % tuple(args[1:]))
                except TypeError:
                    print(*args)

    cfg = load_feature_extract_config(config_path)
    device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    feature_hw = tuple(cfg["dataset"].get("feature_hw") or cfg["dataset"].get("teacher_feature_hw"))
    renderer = MapFeatureRenderer(cfg, feature_hw=feature_hw, device=device_obj, logger=_Logger())
    checkpoint = safe_torch_load(checkpoint_path)
    renderer.load_trainable_state(checkpoint.get("map_renderer_state_dict"))
    renderer.eval()
    with torch.no_grad():
        return extract_rendered_map_coarse_descriptors_from_renderer(renderer, samples)


def search_coarse_pose_bank(
    query_descriptors: torch.Tensor,
    bank_descriptors: torch.Tensor,
    *,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top-k cosine matches from a normalized or unnormalized coarse descriptor bank."""
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


def build_coarse_pose_bank_entries(
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

    topk = int(indices_np.shape[1])
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
        "method_requested": "coarse_pose_bank",
        "method_used": source_name,
        "retrieval_topk_requested": topk,
        "num_query_samples": int(len(entries)),
        "num_train_samples": int(len(train_samples)),
        "counts_by_source": {source_name: int(len(entries))},
    }
    if save_path:
        save_retrieval_init_entries(entries, stats, save_path)
    return entries, stats


def _candidate_valid_mask(entry: Mapping, length: int) -> np.ndarray:
    mask = np.asarray(entry.get("candidate_valid_mask", np.ones((length,), dtype=bool)), dtype=bool).reshape(-1)
    if len(mask) < length:
        padded = np.zeros((length,), dtype=bool)
        padded[: len(mask)] = mask
        return padded
    return mask[:length]


def summarize_topk_pose_recall(
    entries: Sequence[Dict],
    gt_poses_by_name: Mapping[str, np.ndarray],
    *,
    topks: Sequence[int] = (1, 5, 10, 20, 50),
) -> Dict[str, float]:
    """Summarize best valid candidate pose error among the first-k candidates."""
    usable = [entry for entry in entries if str(entry["query_image_name"]) in gt_poses_by_name]
    if not usable:
        raise ValueError("No entries have matching ground-truth poses")

    summary: Dict[str, float] = {"num_samples": int(len(usable))}
    for raw_k in topks:
        k = max(1, int(raw_k))
        best_rot = []
        best_trans = []
        for entry in usable:
            poses = np.asarray(entry.get("pose_init_candidates", np.asarray(entry["pose_init"])[None]), dtype=np.float32)
            if poses.ndim == 2:
                poses = poses[None]
            valid = _candidate_valid_mask(entry, len(poses))
            gt = np.asarray(gt_poses_by_name[str(entry["query_image_name"])], dtype=np.float32)
            candidate_errors = []
            for idx in range(min(k, len(poses))):
                if not bool(valid[idx]):
                    continue
                rot, trans = pose_error(poses[idx], gt)
                candidate_errors.append((float(rot), float(trans)))
            if not candidate_errors:
                rot, trans = pose_error(np.asarray(entry["pose_init"], dtype=np.float32), gt)
                candidate_errors.append((float(rot), float(trans)))
            best = min(candidate_errors, key=lambda item: item[1])
            best_rot.append(best[0])
            best_trans.append(best[1])

        rot_np = np.asarray(best_rot, dtype=np.float64)
        trans_np = np.asarray(best_trans, dtype=np.float64)
        summary[f"top{k}_rot_median"] = float(np.median(rot_np))
        summary[f"top{k}_trans_median"] = float(np.median(trans_np))
        summary[f"top{k}_joint_1deg_100mm"] = float(np.mean((rot_np < 1.0) & (trans_np < 100.0)) * 100.0)
        summary[f"top{k}_joint_5deg_250mm"] = float(np.mean((rot_np < 5.0) & (trans_np < 250.0)) * 100.0)
    return summary


def export_coarse_pose_bank_init(
    *,
    cached_feature_dir: str,
    colmap_dir: str,
    train_split: str,
    query_split: str,
    save_path: str,
    topk: int = 50,
    source_name: str = "coarse_pose_bank",
    summary_path: str | None = None,
    bank_source: str = "query_cache",
    config_path: str | None = None,
    checkpoint_path: str | None = None,
    device: str = "cuda",
) -> Dict:
    train_samples = list_colmap_split_samples(colmap_dir, train_split)
    query_samples = list_colmap_split_samples(colmap_dir, query_split)
    used_query_samples, query_desc = extract_cached_coarse_descriptors(cached_feature_dir, query_samples)
    bank_source_key = str(bank_source or "query_cache").lower()
    if bank_source_key == "query_cache":
        used_train_samples, bank_desc = extract_cached_coarse_descriptors(cached_feature_dir, train_samples)
        descriptor_source = "cached_coarse_sem"
    elif bank_source_key == "map_render":
        if not config_path or not checkpoint_path:
            raise ValueError("--config and --checkpoint are required when --bank_source=map_render")
        used_train_samples, bank_desc = extract_rendered_map_coarse_descriptors(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            samples=train_samples,
            device=device,
        )
        descriptor_source = "rendered_map_coarse_sem"
    else:
        raise ValueError("bank_source must be one of {'query_cache', 'map_render'}")
    indices, scores = search_coarse_pose_bank(query_desc, bank_desc, topk=topk)
    entries, stats = build_coarse_pose_bank_entries(
        query_samples=used_query_samples,
        train_samples=used_train_samples,
        indices=indices,
        scores=scores,
        source_name=source_name,
        save_path=save_path,
    )
    gt_poses = {str(sample["image_name"]): np.asarray(sample["pose_w2c"], dtype=np.float32) for sample in used_query_samples}
    recall = summarize_topk_pose_recall(entries, gt_poses, topks=(1, 5, 10, 20, 50))
    stats.update(
        {
            "descriptor_dim": int(bank_desc.shape[1]),
            "descriptor_source": descriptor_source,
            "bank_source": bank_source_key,
            "num_query_samples_requested": int(len(query_samples)),
            "num_train_samples_requested": int(len(train_samples)),
            "topk_pose_recall": recall,
        }
    )
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export coarse pose-bank real-init cache")
    parser.add_argument("--cached_feature_dir", required=True)
    parser.add_argument("--colmap_dir", required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--query_split", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--source_name", default="coarse_pose_bank")
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--bank_source", choices=["query_cache", "map_render"], default="query_cache")
    parser.add_argument("--config", default=None, help="Feature config for --bank_source=map_render")
    parser.add_argument("--checkpoint", default=None, help="Feature checkpoint for --bank_source=map_render")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = export_coarse_pose_bank_init(
        cached_feature_dir=args.cached_feature_dir,
        colmap_dir=args.colmap_dir,
        train_split=args.train_split,
        query_split=args.query_split,
        save_path=args.save_path,
        topk=args.topk,
        source_name=args.source_name,
        summary_path=args.summary_path,
        bank_source=args.bank_source,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
