"""Reference-pose candidate banks for public localizability benchmarks."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from feature_extract.localizability.candidate_bank import CandidateBank, CandidateBankMetadata


def parse_hloc_pairs_lines(lines: Iterable[str]) -> "OrderedDict[str, list[str]]":
    query_to_refs: "OrderedDict[str, list[str]]" = OrderedDict()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        query, ref = parts[0], parts[1]
        query_to_refs.setdefault(query, []).append(ref)
    return query_to_refs


def parse_hloc_pairs_file(path: str | Path) -> "OrderedDict[str, list[str]]":
    with Path(path).open("r", encoding="utf-8") as handle:
        return parse_hloc_pairs_lines(handle)


def _camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64)
    return (-(pose[:3, :3].T @ pose[:3, 3])).astype(np.float64)


def _pose_errors(candidate_w2c: np.ndarray, gt_w2c: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(candidate_w2c, dtype=np.float64)
    gt = np.asarray(gt_w2c, dtype=np.float64)
    trans = float(np.linalg.norm(_camera_center_from_w2c(pred) - _camera_center_from_w2c(gt)))
    rel = pred[:3, :3].T @ gt[:3, :3]
    cos_angle = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    rot = float(math.degrees(math.acos(cos_angle)))
    return trans, rot


def build_reference_pose_bank(
    *,
    query_poses: Mapping[str, np.ndarray],
    reference_poses: Mapping[str, np.ndarray],
    query_to_refs: Mapping[str, list[str]],
    topk: int,
    scene: str | None = None,
    rot_cost_weight: float = 0.1,
) -> CandidateBank:
    """Build a fixed-width candidate bank from query-reference pose pairs."""
    k = max(1, int(topk))
    sample_names: list[str] = []
    pose_gt = []
    candidate_pose = []
    trans_err = []
    rot_err = []
    pose_cost = []
    valid_mask = []
    reference_names = []

    for query_name, refs in query_to_refs.items():
        if query_name not in query_poses:
            continue
        gt = np.asarray(query_poses[query_name], dtype=np.float32)
        sample_names.append(str(query_name))
        pose_gt.append(gt)
        cand_poses = np.repeat(gt[None], k, axis=0).astype(np.float32)
        cand_trans = np.full((k,), float("inf"), dtype=np.float32)
        cand_rot = np.full((k,), float("inf"), dtype=np.float32)
        cand_cost = np.full((k,), float("inf"), dtype=np.float32)
        cand_valid = np.zeros((k,), dtype=bool)
        cand_ref_names = np.asarray([""] * k, dtype=object)

        write_idx = 0
        for ref_name in refs:
            if write_idx >= k:
                break
            if ref_name not in reference_poses:
                continue
            ref_pose = np.asarray(reference_poses[ref_name], dtype=np.float32)
            t_err, r_err = _pose_errors(ref_pose, gt)
            cand_poses[write_idx] = ref_pose
            cand_trans[write_idx] = np.float32(t_err)
            cand_rot[write_idx] = np.float32(r_err)
            cand_cost[write_idx] = np.float32(t_err + float(rot_cost_weight) * math.radians(r_err))
            cand_valid[write_idx] = True
            cand_ref_names[write_idx] = str(ref_name)
            write_idx += 1

        candidate_pose.append(cand_poses)
        trans_err.append(cand_trans)
        rot_err.append(cand_rot)
        pose_cost.append(cand_cost)
        valid_mask.append(cand_valid)
        reference_names.append(cand_ref_names)

    if not sample_names:
        raise ValueError("No query poses matched the provided reference-pose pairs")

    bank = CandidateBank(
        sample_names=sample_names,
        pose_gt=torch.as_tensor(np.stack(pose_gt), dtype=torch.float32),
        candidate_pose=torch.as_tensor(np.stack(candidate_pose), dtype=torch.float32),
        pose_cost_m=torch.as_tensor(np.stack(pose_cost), dtype=torch.float32),
        trans_err_m=torch.as_tensor(np.stack(trans_err), dtype=torch.float32),
        rot_err_deg=torch.as_tensor(np.stack(rot_err), dtype=torch.float32),
        valid_mask=torch.as_tensor(np.stack(valid_mask), dtype=torch.bool),
        metadata=CandidateBankMetadata(
            scene=scene,
            candidate_source="reference_pose_pairs",
            extras={"reference_names": reference_names, "rot_cost_weight": float(rot_cost_weight)},
        ),
    )
    return bank


def save_reference_pose_bank(bank: CandidateBank, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference_names = bank.metadata.extras.get("reference_names")
    np.savez_compressed(
        path,
        sample_names=np.asarray(bank.sample_names),
        pose_gt=bank.pose_gt.cpu().numpy().astype(np.float32),
        candidates=bank.candidate_pose.cpu().numpy().astype(np.float32),
        pose_cost_m=bank.pose_cost_m.cpu().numpy().astype(np.float32),
        trans_err_m=bank.trans_err_m.cpu().numpy().astype(np.float32),
        rot_err_deg=bank.rot_err_deg.cpu().numpy().astype(np.float32),
        valid_mask=bank.valid_mask.cpu().numpy().astype(bool) if bank.valid_mask is not None else None,
        reference_names=np.stack(reference_names) if reference_names is not None else None,
        scene=np.asarray(bank.metadata.scene or ""),
        candidate_source=np.asarray(bank.metadata.candidate_source or "reference_pose_pairs"),
        rot_cost_weight=np.asarray(float(bank.metadata.extras.get("rot_cost_weight", 0.1)), dtype=np.float32),
    )
