"""Guarded pose-refinement policies for localization init caches."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

import numpy as np


def _camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return (-rotation.T @ translation).astype(np.float64)


def pose_delta_trans_rot(reference_w2c: np.ndarray, candidate_w2c: np.ndarray) -> tuple[float, float]:
    """Return camera-center translation delta in meters and rotation delta in degrees."""
    reference = np.asarray(reference_w2c, dtype=np.float64)
    candidate = np.asarray(candidate_w2c, dtype=np.float64)
    if reference.shape != (4, 4) or candidate.shape != (4, 4):
        raise ValueError("poses must have shape (4,4)")
    trans_m = float(np.linalg.norm(_camera_center_from_w2c(reference) - _camera_center_from_w2c(candidate)))
    rel = candidate[:3, :3] @ reference[:3, :3].T
    cos_angle = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    rot_deg = float(np.rad2deg(np.arccos(cos_angle)))
    return trans_m, rot_deg


def _metadata_value(metadata: Mapping[str, object], key: str, default: object) -> object:
    value = metadata.get(key, default)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _copy_guarded_entry(source_entry: Mapping, identity_entry: Mapping, init_source: str) -> dict:
    entry = deepcopy(dict(source_entry))
    entry["query_img_id"] = int(identity_entry.get("query_img_id", entry.get("query_img_id", -1)))
    entry["query_image_name"] = str(identity_entry["query_image_name"])
    entry["query_image_stem"] = str(identity_entry.get("query_image_stem", entry.get("query_image_stem", "")))
    entry["init_source"] = str(init_source)
    return entry


def build_guarded_refinement_entries(
    identity_entries: Sequence[Mapping],
    refined_entries: Sequence[Mapping],
    *,
    refinement_metadata: Mapping[str, Mapping[str, object]] | None = None,
    min_inliers: int = 100,
    max_delta_trans_m: float = 0.35,
    max_delta_rot_deg: float = 5.0,
    accepted_source: str = "guarded_render_loftr_refine",
    fallback_source: str = "guarded_identity_fallback",
) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    """Select refined poses only when solver and pose-delta guards pass.

    The identity cache defines query order and fallback poses. The refined cache
    may contain a subset or reordered copy of those queries. ``refinement_metadata``
    is keyed by query image name and can provide ``refine_success`` and
    ``refine_num_inliers`` values loaded from a solver cache.
    """
    if min_inliers < 0:
        raise ValueError("min_inliers must be non-negative")
    refined_by_name = {str(entry["query_image_name"]): entry for entry in refined_entries}
    metadata_by_name = refinement_metadata or {}
    guarded_entries: list[dict] = []
    names: list[str] = []
    accepted_mask: list[bool] = []
    delta_trans: list[float] = []
    delta_rot: list[float] = []
    inliers_values: list[float] = []
    success_values: list[bool] = []
    reasons: list[str] = []
    stats = {
        "policy": "guarded_refinement",
        "num_entries": int(len(identity_entries)),
        "accepted_refined": 0,
        "fallback_identity": 0,
        "reject_missing_refined": 0,
        "reject_solver_failed": 0,
        "reject_low_inliers": 0,
        "reject_large_delta": 0,
        "min_inliers": int(min_inliers),
        "max_delta_trans_m": float(max_delta_trans_m),
        "max_delta_rot_deg": float(max_delta_rot_deg),
        "counts_by_source": {},
    }

    for identity_entry in identity_entries:
        name = str(identity_entry["query_image_name"])
        refined_entry = refined_by_name.get(name)
        metadata = metadata_by_name.get(name, {})
        names.append(name)
        accepted = False
        reason = "accepted"
        trans_m = float("nan")
        rot_deg = float("nan")
        success = bool(_metadata_value(metadata, "refine_success", True))
        inliers = float(_metadata_value(metadata, "refine_num_inliers", float("inf")))

        if refined_entry is None:
            reason = "missing_refined"
            stats["reject_missing_refined"] += 1
        else:
            trans_m, rot_deg = pose_delta_trans_rot(identity_entry["pose_init"], refined_entry["pose_init"])
            if not success:
                reason = "solver_failed"
                stats["reject_solver_failed"] += 1
            elif inliers < float(min_inliers):
                reason = "low_inliers"
                stats["reject_low_inliers"] += 1
            elif trans_m > float(max_delta_trans_m) or rot_deg > float(max_delta_rot_deg):
                reason = "large_delta"
                stats["reject_large_delta"] += 1
            else:
                accepted = True

        if accepted:
            assert refined_entry is not None
            entry = _copy_guarded_entry(refined_entry, identity_entry, accepted_source)
            stats["accepted_refined"] += 1
        else:
            entry = _copy_guarded_entry(identity_entry, identity_entry, fallback_source)
            stats["fallback_identity"] += 1

        source = str(entry.get("init_source", ""))
        stats["counts_by_source"][source] = stats["counts_by_source"].get(source, 0) + 1
        guarded_entries.append(entry)
        accepted_mask.append(bool(accepted))
        delta_trans.append(float(trans_m))
        delta_rot.append(float(rot_deg))
        inliers_values.append(float(inliers))
        success_values.append(bool(success))
        reasons.append(reason)

    diagnostics = {
        "query_image_names": np.asarray(names),
        "accepted_mask": np.asarray(accepted_mask, dtype=bool),
        "delta_trans_m": np.asarray(delta_trans, dtype=np.float32),
        "delta_rot_deg": np.asarray(delta_rot, dtype=np.float32),
        "refine_success": np.asarray(success_values, dtype=bool),
        "refine_num_inliers": np.asarray(inliers_values, dtype=np.float32),
        "guard_reason": np.asarray(reasons),
    }
    return guarded_entries, stats, diagnostics
