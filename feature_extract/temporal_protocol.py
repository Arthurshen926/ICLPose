"""Temporal localization protocol helpers for Cambridge-style sequences."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch


_SEQ_RE = re.compile(r"seq(\d+)", re.IGNORECASE)
_FRAME_RE = re.compile(r"frame(\d+)", re.IGNORECASE)


def parse_sequence_frame(sample_name: str) -> tuple[str, int]:
    """Return the sequence id and numeric frame id from a Cambridge sample name."""
    normalized = str(sample_name).replace("\\", "/")
    parts = normalized.split("/")
    seq = parts[-2] if len(parts) >= 2 else ""
    frame_stem = Path(parts[-1]).stem
    frame_match = _FRAME_RE.search(frame_stem)
    if frame_match is None:
        raise ValueError(f"Could not parse frame id from sample name '{sample_name}'")
    return seq, int(frame_match.group(1))


def _sequence_sort_key(seq: str) -> tuple[int, str]:
    match = _SEQ_RE.search(seq)
    if match is None:
        return (10**9, seq)
    return (int(match.group(1)), seq)


def temporal_sort_records(records: Iterable[dict]) -> list[dict]:
    """Sort records by sequence and numeric frame without mutating the input."""
    return sorted(
        list(records),
        key=lambda record: (
            _sequence_sort_key(parse_sequence_frame(record["sample_name"])[0]),
            parse_sequence_frame(record["sample_name"])[1],
            record["sample_name"],
        ),
    )


def camera_center_from_w2c(pose_w2c: torch.Tensor) -> torch.Tensor:
    """Compute camera center in world coordinates from a world-to-camera pose."""
    pose = pose_w2c.float()
    rotation = pose[..., :3, :3]
    translation = pose[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)


def w2c_with_camera_center(template_w2c: torch.Tensor, center_world: torch.Tensor) -> torch.Tensor:
    """Keep template rotation and replace its camera center."""
    pose = template_w2c.clone()
    rotation = pose[:3, :3]
    center = center_world.to(device=pose.device, dtype=pose.dtype)
    pose[:3, 3] = -(rotation @ center)
    return pose


def constant_velocity_w2c(previous_previous: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Predict the next pose center with constant velocity and keep previous rotation."""
    center_0 = camera_center_from_w2c(previous_previous)
    center_1 = camera_center_from_w2c(previous)
    center_next = center_1 + (center_1 - center_0)
    return w2c_with_camera_center(previous, center_next)


def _yaw_rotation_world(angle_deg: float, *, device, dtype) -> torch.Tensor:
    angle = torch.tensor(float(angle_deg), device=device, dtype=dtype) * torch.pi / 180.0
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rotation = torch.eye(3, device=device, dtype=dtype)
    rotation[0, 0] = cos_a
    rotation[0, 2] = sin_a
    rotation[2, 0] = -sin_a
    rotation[2, 2] = cos_a
    return rotation


def build_temporal_pose_grid(
    base_pose_w2c: torch.Tensor,
    *,
    trans_offsets_m: Iterable[float],
    yaw_offsets_deg: Iterable[float],
) -> torch.Tensor:
    """Build a local pose grid around a temporal initializer.

    Translation offsets are sampled in the world X/Z plane. Yaw is applied around
    the world Y axis while preserving the sampled camera center.
    """
    base = base_pose_w2c.float()
    center = camera_center_from_w2c(base)
    base_rotation = base[:3, :3]
    candidates = []
    for dx in trans_offsets_m:
        for dz in trans_offsets_m:
            center_i = center + torch.tensor([float(dx), 0.0, float(dz)], device=base.device, dtype=base.dtype)
            for yaw in yaw_offsets_deg:
                yaw_world = _yaw_rotation_world(float(yaw), device=base.device, dtype=base.dtype)
                pose_i = base.clone()
                pose_i[:3, :3] = base_rotation @ yaw_world.transpose(0, 1)
                pose_i[:3, 3] = -(pose_i[:3, :3] @ center_i)
                candidates.append(pose_i)
    if not candidates:
        raise ValueError("Temporal pose grid needs at least one translation and yaw offset")
    return torch.stack(candidates, dim=0)


class TemporalInitSelector:
    """Stateful initializer selector for single-frame and temporal protocols."""

    def __init__(self, protocol: str = "single_frame_real_init", init_mode: str = "prev_pose"):
        self.protocol = str(protocol)
        self.init_mode = str(init_mode)
        if self.protocol not in {
            "single_frame_real_init",
            "temporal_prev_gt_oracle",
            "temporal_prev_pred_tracking",
        }:
            raise ValueError(f"Unknown temporal protocol '{protocol}'")
        if self.init_mode not in {"prev_pose", "constant_velocity"}:
            raise ValueError(f"Unknown temporal init mode '{init_mode}'")
        self.current_sequence: str | None = None
        self.prev_gt: torch.Tensor | None = None
        self.prev_prev_gt: torch.Tensor | None = None
        self.prev_pred: torch.Tensor | None = None
        self.prev_prev_pred: torch.Tensor | None = None

    def _select_temporal_source(self) -> tuple[torch.Tensor | None, str | None]:
        if self.protocol == "temporal_prev_gt_oracle":
            if self.init_mode == "constant_velocity" and self.prev_prev_gt is not None and self.prev_gt is not None:
                return constant_velocity_w2c(self.prev_prev_gt, self.prev_gt), "constant_velocity_gt"
            if self.prev_gt is not None:
                return self.prev_gt, "prev_gt"
            return None, None
        if self.protocol == "temporal_prev_pred_tracking":
            if self.init_mode == "constant_velocity" and self.prev_prev_pred is not None and self.prev_pred is not None:
                return constant_velocity_w2c(self.prev_prev_pred, self.prev_pred), "constant_velocity_pred"
            if self.prev_pred is not None:
                return self.prev_pred, "prev_pred"
            return None, None
        return None, None

    def select(self, sample_name: str, real_init_pose: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Select the initial pose for a sample and return metadata for logging."""
        seq, frame = parse_sequence_frame(sample_name)
        is_sequence_start = self.current_sequence != seq
        if self.protocol == "single_frame_real_init" or is_sequence_start:
            return real_init_pose.clone(), {
                "sequence": seq,
                "frame": frame,
                "is_sequence_start": True,
                "source": "real_init",
            }
        pose, source = self._select_temporal_source()
        if pose is None:
            return real_init_pose.clone(), {
                "sequence": seq,
                "frame": frame,
                "is_sequence_start": True,
                "source": "real_init",
            }
        return pose.clone(), {
            "sequence": seq,
            "frame": frame,
            "is_sequence_start": False,
            "source": source,
        }

    def update(self, sample_name: str, *, gt_pose: torch.Tensor, final_pose: torch.Tensor) -> None:
        """Update sequence-local history after processing a sample."""
        seq, _frame = parse_sequence_frame(sample_name)
        if self.current_sequence != seq:
            self.current_sequence = seq
            self.prev_gt = None
            self.prev_prev_gt = None
            self.prev_pred = None
            self.prev_prev_pred = None
        self.prev_prev_gt = self.prev_gt
        self.prev_gt = gt_pose.detach().clone()
        self.prev_prev_pred = self.prev_pred
        self.prev_pred = final_pose.detach().clone()


def summarize_error_pairs(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """Summarize translation/rotation errors with fixed localization recalls."""
    if not pairs:
        return {}
    trans = torch.tensor([pair[0] for pair in pairs], dtype=torch.float32)
    rot = torch.tensor([pair[1] for pair in pairs], dtype=torch.float32)
    return {
        "mean_trans_mm": float(trans.mean().item()),
        "median_trans_mm": float(trans.median().item()),
        "mean_rot_deg": float(rot.mean().item()),
        "median_rot_deg": float(rot.median().item()),
        "recall_1deg_50mm": float(((rot < 1.0) & (trans < 50.0)).float().mean().item() * 100.0),
        "recall_1deg_100mm": float(((rot < 1.0) & (trans < 100.0)).float().mean().item() * 100.0),
        "recall_2deg_100mm": float(((rot < 2.0) & (trans < 100.0)).float().mean().item() * 100.0),
        "recall_5deg_250mm": float(((rot < 5.0) & (trans < 250.0)).float().mean().item() * 100.0),
    }


def summarize_temporal_errors(samples: list[dict]) -> dict[str, dict[str, float]]:
    """Summarize init/final temporal samples and non-start-frame subsets."""
    init_pairs = [(float(sample["init_trans_mm"]), float(sample["init_rot_deg"])) for sample in samples]
    final_pairs = [(float(sample["final_trans_mm"]), float(sample["final_rot_deg"])) for sample in samples]
    non_start = [sample for sample in samples if not bool(sample.get("is_sequence_start", False))]
    init_non_start = [(float(sample["init_trans_mm"]), float(sample["init_rot_deg"])) for sample in non_start]
    final_non_start = [(float(sample["final_trans_mm"]), float(sample["final_rot_deg"])) for sample in non_start]
    if samples:
        trans_gain = torch.tensor(
            [float(sample["init_trans_mm"]) - float(sample["final_trans_mm"]) for sample in samples],
            dtype=torch.float32,
        )
        rot_gain = torch.tensor(
            [float(sample["init_rot_deg"]) - float(sample["final_rot_deg"]) for sample in samples],
            dtype=torch.float32,
        )
        gain = {
            "mean_trans_mm": float(trans_gain.mean().item()),
            "median_trans_mm": float(trans_gain.median().item()),
            "mean_rot_deg": float(rot_gain.mean().item()),
            "median_rot_deg": float(rot_gain.median().item()),
        }
    else:
        gain = {}
    return {
        "init": summarize_error_pairs(init_pairs),
        "final": summarize_error_pairs(final_pairs),
        "init_non_start": summarize_error_pairs(init_non_start),
        "final_non_start": summarize_error_pairs(final_non_start),
        "init_to_final_gain": gain,
    }
