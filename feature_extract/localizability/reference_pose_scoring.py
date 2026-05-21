"""Feature scoring utilities for reference-pose localizability banks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


def descriptor_from_dense_feature(feature: torch.Tensor) -> torch.Tensor:
    """Pool a dense CxHxW or HxWxC feature into one L2-normalized descriptor."""
    feat = feature.detach().float().cpu()
    if feat.ndim != 3:
        raise ValueError("dense feature must have 3 dimensions")
    if feat.shape[0] <= feat.shape[-1]:
        channels_first = feat
    else:
        channels_first = feat.permute(2, 0, 1).contiguous()
    desc = channels_first.reshape(channels_first.shape[0], -1).mean(dim=1)
    return F.normalize(desc, dim=0, eps=1.0e-6)


def score_reference_pose_descriptors(
    *,
    sample_names: Sequence[str],
    reference_names: Sequence[Sequence[str]],
    descriptors: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score query/reference candidates by cosine descriptor similarity."""
    if len(sample_names) != len(reference_names):
        raise ValueError("sample_names and reference_names must have the same length")
    num_candidates = max((len(refs) for refs in reference_names), default=0)
    if num_candidates <= 0:
        raise ValueError("reference_names must contain at least one candidate")
    scores = torch.full((len(sample_names), num_candidates), -1.0e9, dtype=torch.float32)
    valid = torch.zeros((len(sample_names), num_candidates), dtype=torch.bool)
    for row_idx, (sample_name, refs) in enumerate(zip(sample_names, reference_names)):
        q_desc = descriptors.get(str(sample_name))
        if q_desc is None:
            continue
        q = F.normalize(q_desc.float().view(-1), dim=0, eps=1.0e-6)
        for cand_idx, ref_name in enumerate(refs):
            if cand_idx >= num_candidates:
                break
            if not str(ref_name):
                continue
            r_desc = descriptors.get(str(ref_name))
            if r_desc is None:
                continue
            r = F.normalize(r_desc.float().view(-1), dim=0, eps=1.0e-6)
            if r.shape != q.shape:
                continue
            scores[row_idx, cand_idx] = float((q * r).sum())
            valid[row_idx, cand_idx] = True
    return scores, valid


def retrieval_order_scores(valid_mask: torch.Tensor) -> torch.Tensor:
    """Return deterministic scores that preserve candidate-list order."""
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape (B,K)")
    bsz, num_candidates = valid_mask.shape
    ranks = torch.arange(num_candidates, dtype=torch.float32, device=valid_mask.device)
    scores = -ranks.view(1, num_candidates).expand(bsz, num_candidates).clone()
    return scores.masked_fill(~valid_mask.bool(), -1.0e9)


def save_descriptor_bank(
    path: str | Path,
    descriptors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping | None = None,
) -> None:
    """Save a compact sample-name keyed descriptor bank."""
    names = sorted(str(name) for name in descriptors)
    vectors = []
    for name in names:
        vec = F.normalize(descriptors[name].detach().float().view(-1).cpu(), dim=0, eps=1.0e-6)
        vectors.append(vec)
    if vectors:
        dim = int(vectors[0].numel())
        if any(int(vec.numel()) != dim for vec in vectors):
            raise ValueError("all descriptors must have the same dimension")
        tensor = torch.stack(vectors, dim=0)
    else:
        tensor = torch.empty((0, 0), dtype=torch.float32)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "names": names,
            "descriptors": tensor,
            "metadata": dict(metadata or {}),
        },
        out_path,
    )


def load_descriptor_bank(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a descriptor bank saved by :func:`save_descriptor_bank`."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    names = [str(name) for name in payload.get("names", [])]
    tensor = payload.get("descriptors")
    if tensor is None:
        raise ValueError("descriptor bank missing 'descriptors'")
    tensor = tensor.float().cpu()
    if tensor.ndim != 2 or tensor.shape[0] != len(names):
        raise ValueError("descriptor bank shape does not match names")
    descriptors = {
        name: F.normalize(tensor[idx].float(), dim=0, eps=1.0e-6)
        for idx, name in enumerate(names)
    }
    metadata = dict(payload.get("metadata", {}))
    return descriptors, metadata


def find_feature_path(feature_root: str | Path, image_id: int, *, subdir: str = "fine_geo") -> Path | None:
    root = Path(feature_root) / str(subdir)
    if not root.is_dir():
        return None
    matches = sorted(root.glob(f"rgb_{int(image_id)}_*x*x*.pt"))
    if not matches:
        matches = sorted(root.glob(f"rgb_{int(image_id)}_*.pt"))
    return matches[0] if matches else None


def load_descriptors_for_names(
    *,
    names: Sequence[str],
    name_to_image_id: Mapping[str, int],
    feature_root: str | Path,
    subdir: str = "fine_geo",
) -> dict[str, torch.Tensor]:
    descriptors: dict[str, torch.Tensor] = {}
    for name in sorted({str(v) for v in names if str(v)}):
        image_id = name_to_image_id.get(name)
        if image_id is None:
            continue
        path = find_feature_path(feature_root, image_id, subdir=subdir)
        if path is None:
            continue
        try:
            feature = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            feature = torch.load(path, map_location="cpu")
        descriptors[name] = descriptor_from_dense_feature(feature)
    return descriptors
