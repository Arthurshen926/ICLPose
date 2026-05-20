"""Candidate-bank helpers for POFD-FS controlled and stress-test ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CandidateBankMetadata:
    source_path: str | None = None
    scene: str | None = None
    candidate_source: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateBank:
    sample_names: list[str]
    pose_gt: torch.Tensor
    candidate_pose: torch.Tensor
    pose_cost_m: torch.Tensor
    trans_err_m: torch.Tensor
    rot_err_deg: torch.Tensor
    valid_mask: torch.Tensor | None = None
    metadata: CandidateBankMetadata = field(default_factory=CandidateBankMetadata)

    def __post_init__(self) -> None:
        if self.candidate_pose.ndim != 4 or self.candidate_pose.shape[-2:] != (4, 4):
            raise ValueError("candidate_pose must have shape (B,K,4,4)")
        bsz, num_candidates = self.candidate_pose.shape[:2]
        if self.pose_gt.shape != (bsz, 4, 4):
            raise ValueError("pose_gt must have shape (B,4,4)")
        for name, tensor in {
            "pose_cost_m": self.pose_cost_m,
            "trans_err_m": self.trans_err_m,
            "rot_err_deg": self.rot_err_deg,
        }.items():
            if tensor.shape != (bsz, num_candidates):
                raise ValueError(f"{name} must have shape (B,K)")
        if self.valid_mask is not None and self.valid_mask.shape != (bsz, num_candidates):
            raise ValueError("valid_mask must have shape (B,K)")
        if len(self.sample_names) != bsz:
            raise ValueError("sample_names length must match batch size")

    def basin_label(self, trans_thresh_m: float, rot_thresh_deg: float) -> torch.Tensor:
        valid = self.valid_mask if self.valid_mask is not None else torch.ones_like(self.trans_err_m, dtype=torch.bool)
        return (self.trans_err_m <= float(trans_thresh_m)) & (self.rot_err_deg <= float(rot_thresh_deg)) & valid.bool()


def _first_key(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in data:
            return key
    return None


def _tensor(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], *, dtype=torch.float32) -> torch.Tensor:
    key = _first_key(data, keys)
    if key is None:
        raise KeyError(f"None of these keys were found in candidate bank: {keys}")
    return torch.as_tensor(data[key], dtype=dtype)


def _sample_names(data: np.lib.npyio.NpzFile, bsz: int) -> list[str]:
    key = _first_key(data, ("sample_names", "image_names", "query_names", "names"))
    if key is None:
        return [f"sample_{idx:06d}" for idx in range(bsz)]
    return [str(value) for value in data[key].tolist()]


def candidate_bank_from_npz(path: str | Path) -> CandidateBank:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        candidate_pose = _tensor(data, ("candidates", "candidate_pose", "candidate_poses", "pose_init_candidates"))
        pose_gt = _tensor(data, ("pose_gt", "poses_gt", "gt_pose", "gt_poses"))
        bsz, num_candidates = candidate_pose.shape[:2]
        pose_cost_key = _first_key(data, ("pose_cost_m", "candidate_cost_m", "cost_m"))
        trans_key = _first_key(data, ("trans_err_m", "candidate_trans_err_m", "trans_errors_m"))
        rot_key = _first_key(data, ("rot_err_deg", "candidate_rot_err_deg", "rot_errors_deg"))
        trans_err = torch.as_tensor(data[trans_key], dtype=torch.float32) if trans_key else torch.zeros(bsz, num_candidates)
        rot_err = torch.as_tensor(data[rot_key], dtype=torch.float32) if rot_key else torch.zeros(bsz, num_candidates)
        pose_cost = (
            torch.as_tensor(data[pose_cost_key], dtype=torch.float32)
            if pose_cost_key
            else trans_err + 0.1 * torch.deg2rad(rot_err)
        )
        valid_key = _first_key(data, ("valid_mask", "candidate_valid_mask", "valid"))
        valid_mask = torch.as_tensor(data[valid_key], dtype=torch.bool) if valid_key else torch.ones_like(pose_cost, dtype=torch.bool)
        metadata = CandidateBankMetadata(
            source_path=str(path),
            scene=str(data["scene"].tolist()) if "scene" in data else None,
            candidate_source=str(data["candidate_source"].tolist()) if "candidate_source" in data else None,
            extras={"keys": list(data.files)},
        )
        return CandidateBank(
            sample_names=_sample_names(data, bsz),
            pose_gt=pose_gt,
            candidate_pose=candidate_pose,
            pose_cost_m=pose_cost,
            trans_err_m=trans_err,
            rot_err_deg=rot_err,
            valid_mask=valid_mask,
            metadata=metadata,
        )
