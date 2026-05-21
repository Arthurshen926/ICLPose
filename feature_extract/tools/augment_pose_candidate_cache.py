#!/usr/bin/env python3
"""Append identity/near-init pose candidates to an existing candidate cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pose_refine import apply_pose_delta  # noqa: E402


def _parse_float_csv(value: str) -> list[float]:
    parsed = [float(part) for part in str(value).replace(";", ",").split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected at least one numeric CSV value")
    return parsed


def build_near_identity_pose_candidates(
    pose_inits: np.ndarray,
    *,
    trans_cm: list[float] | tuple[float, ...],
    rot_deg: list[float] | tuple[float, ...],
    num_jitter: int,
    seed: int,
    include_identity: bool = True,
) -> np.ndarray:
    pose = torch.as_tensor(pose_inits, dtype=torch.float32)
    if pose.ndim != 3 or pose.shape[-2:] != (4, 4):
        raise ValueError("pose_inits must have shape (B,4,4)")
    bsz = int(pose.shape[0])
    rows = []
    if include_identity:
        rows.append(pose[:, None])
    if int(num_jitter) > 0:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        trans_choices = torch.as_tensor(trans_cm, dtype=torch.float32) / 100.0
        rot_choices = torch.deg2rad(torch.as_tensor(rot_deg, dtype=torch.float32))
        trans_idx = torch.randint(0, len(trans_choices), (bsz, int(num_jitter)), generator=generator)
        rot_idx = torch.randint(0, len(rot_choices), (bsz, int(num_jitter)), generator=generator)
        trans_mag = trans_choices[trans_idx]
        rot_mag = rot_choices[rot_idx]
        trans_dir = torch.randn(bsz, int(num_jitter), 3, generator=generator)
        trans_dir = trans_dir / trans_dir.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        rot_dir = torch.randn(bsz, int(num_jitter), 3, generator=generator)
        rot_dir = rot_dir / rot_dir.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        delta = torch.zeros(bsz, int(num_jitter), 6, dtype=torch.float32)
        delta[..., :3] = trans_dir * trans_mag.unsqueeze(-1)
        delta[..., 3:] = rot_dir * rot_mag.unsqueeze(-1)
        jittered = apply_pose_delta(
            pose[:, None].expand(-1, int(num_jitter), -1, -1).reshape(-1, 4, 4),
            delta.reshape(-1, 6),
        ).reshape(bsz, int(num_jitter), 4, 4)
        rows.append(jittered)
    if not rows:
        raise ValueError("No candidates requested; enable identity or num_jitter > 0")
    return torch.cat(rows, dim=1).cpu().numpy().astype(np.float32)


def _base_candidate_values(cache: dict[str, np.ndarray], key: str, added_count: int) -> np.ndarray | None:
    base_map = {
        "retrieval_frame_ids_candidates": "retrieval_frame_ids",
        "retrieval_image_names_candidates": "retrieval_image_names",
        "retrieval_scores_candidates": "retrieval_scores",
        "retrieval_original_scores_candidates": "retrieval_scores",
    }
    base_key = base_map.get(key)
    if base_key is None or base_key not in cache:
        return None
    base = np.asarray(cache[base_key])
    return np.repeat(base[:, None], added_count, axis=1)


def _default_append_values(cache: dict[str, np.ndarray], key: str, arr: np.ndarray, added_count: int) -> np.ndarray:
    base = _base_candidate_values(cache, key, added_count)
    if base is not None:
        return base.astype(arr.dtype, copy=False)
    bsz = int(arr.shape[0])
    tail_shape = arr.shape[2:]
    if arr.dtype == np.dtype(bool):
        return np.ones((bsz, added_count, *tail_shape), dtype=arr.dtype)
    if np.issubdtype(arr.dtype, np.integer):
        fill = -1 if key == "candidate_permutation" else 0
        return np.full((bsz, added_count, *tail_shape), fill, dtype=arr.dtype)
    if np.issubdtype(arr.dtype, np.floating):
        return np.zeros((bsz, added_count, *tail_shape), dtype=arr.dtype)
    return np.full((bsz, added_count, *tail_shape), "", dtype=arr.dtype)


def append_near_identity_candidate_arrays(
    cache: dict[str, np.ndarray],
    *,
    trans_cm: list[float] | tuple[float, ...],
    rot_deg: list[float] | tuple[float, ...],
    num_jitter: int,
    seed: int,
    include_identity: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    pose_inits = np.asarray(cache["pose_inits"], dtype=np.float32)
    existing = np.asarray(cache["pose_init_candidates"], dtype=np.float32)
    if existing.ndim != 4 or existing.shape[-2:] != (4, 4):
        raise ValueError("pose_init_candidates must have shape (B,K,4,4)")
    added = build_near_identity_pose_candidates(
        pose_inits,
        trans_cm=trans_cm,
        rot_deg=rot_deg,
        num_jitter=int(num_jitter),
        seed=int(seed),
        include_identity=bool(include_identity),
    )
    if added.shape[0] != existing.shape[0]:
        raise ValueError("pose_inits and pose_init_candidates batch sizes differ")

    bsz, old_count = existing.shape[:2]
    added_count = int(added.shape[1])
    augmented: dict[str, np.ndarray] = {}
    for key, value in cache.items():
        arr = np.asarray(value)
        if key == "pose_init_candidates":
            augmented[key] = np.concatenate([existing, added], axis=1)
        elif arr.ndim >= 2 and arr.shape[0] == bsz and arr.shape[1] == old_count:
            tail = _default_append_values(cache, key, arr, added_count)
            augmented[key] = np.concatenate([arr, tail], axis=1)
        else:
            augmented[key] = arr

    metadata = {
        "near_identity_augmented": True,
        "num_existing_candidates": int(old_count),
        "num_added_candidates": int(added_count),
        "num_total_candidates": int(old_count + added_count),
        "include_identity": bool(include_identity),
        "num_jitter": int(num_jitter),
        "trans_cm": [float(v) for v in trans_cm],
        "rot_deg": [float(v) for v in rot_deg],
        "seed": int(seed),
    }
    augmented["stats"] = np.array([metadata], dtype=object)
    return augmented, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trans-cm", default="2.5,5,10")
    parser.add_argument("--rot-deg", default="0.5,1,2")
    parser.add_argument("--num-jitter", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--include-identity", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as data:
        cache = {key: data[key] for key in data.files}
    augmented, metadata = append_near_identity_candidate_arrays(
        cache,
        trans_cm=_parse_float_csv(args.trans_cm),
        rot_deg=_parse_float_csv(args.rot_deg),
        num_jitter=int(args.num_jitter),
        seed=int(args.seed),
        include_identity=bool(args.include_identity),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **augmented)
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metadata, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
