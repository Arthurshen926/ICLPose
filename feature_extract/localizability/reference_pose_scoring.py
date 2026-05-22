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


def patch_descriptors_from_dense_feature(
    feature: torch.Tensor,
    *,
    grid_hw: tuple[int, int] = (4, 4),
) -> torch.Tensor:
    """Pool a dense feature into an L2-normalized grid of patch descriptors."""
    feat = feature.detach().float().cpu()
    if feat.ndim != 3:
        raise ValueError("dense feature must have 3 dimensions")
    if feat.shape[0] <= feat.shape[-1]:
        channels_first = feat
    else:
        channels_first = feat.permute(2, 0, 1).contiguous()
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_hw must be positive")
    pooled = F.adaptive_avg_pool2d(channels_first[None], output_size=(grid_h, grid_w))[0]
    patches = pooled.permute(1, 2, 0).reshape(grid_h * grid_w, channels_first.shape[0])
    return F.normalize(patches, dim=1, eps=1.0e-6)


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


def project_descriptor_bank_pca(
    descriptors: Mapping[str, torch.Tensor],
    *,
    out_dim: int = 64,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Project a descriptor bank to a compact PCA subspace."""
    names = sorted(str(name) for name in descriptors)
    if not names:
        return {}, {"method": "pca", "input_dim": 0, "output_dim": 0, "num_descriptors": 0}
    vectors = [descriptors[name].detach().float().view(-1).cpu() for name in names]
    input_dim = int(vectors[0].numel())
    if any(int(vec.numel()) != input_dim for vec in vectors):
        raise ValueError("all descriptors must have the same dimension")
    matrix = torch.stack(vectors, dim=0)
    dim = max(1, min(int(out_dim), input_dim, int(matrix.shape[0])))
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:dim].t().contiguous()
    projected = F.normalize(centered @ basis, dim=1, eps=1.0e-6)
    return {
        name: projected[idx].float().cpu()
        for idx, name in enumerate(names)
    }, {
        "method": "pca",
        "input_dim": input_dim,
        "output_dim": dim,
        "num_descriptors": len(names),
    }


def _patch_match_score(q: torch.Tensor, r: torch.Tensor, *, topk: int) -> torch.Tensor:
    if q.ndim != 2 or r.ndim != 2:
        raise ValueError("patch descriptors must have shape (P,C)")
    if q.shape[1] != r.shape[1]:
        raise ValueError("query and reference patch descriptors must share channel dimension")
    q_norm = F.normalize(q.float(), dim=1, eps=1.0e-6)
    r_norm = F.normalize(r.float(), dim=1, eps=1.0e-6)
    sim = q_norm @ r_norm.t()
    q_to_r = sim.max(dim=1).values
    r_to_q = sim.max(dim=0).values
    evidence = torch.cat([q_to_r, r_to_q], dim=0)
    k = max(1, min(int(topk), int(evidence.numel())))
    return evidence.topk(k, largest=True).values.mean()


def score_reference_pose_patch_descriptors(
    *,
    sample_names: Sequence[str],
    reference_names: Sequence[Sequence[str]],
    patch_descriptors: Mapping[str, torch.Tensor],
    topk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score query/reference candidates by mutual top-k patch similarity."""
    if len(sample_names) != len(reference_names):
        raise ValueError("sample_names and reference_names must have the same length")
    num_candidates = max((len(refs) for refs in reference_names), default=0)
    if num_candidates <= 0:
        raise ValueError("reference_names must contain at least one candidate")
    scores = torch.full((len(sample_names), num_candidates), -1.0e9, dtype=torch.float32)
    valid = torch.zeros((len(sample_names), num_candidates), dtype=torch.bool)
    for row_idx, (sample_name, refs) in enumerate(zip(sample_names, reference_names)):
        q_desc = patch_descriptors.get(str(sample_name))
        if q_desc is None:
            continue
        for cand_idx, ref_name in enumerate(refs):
            if cand_idx >= num_candidates:
                break
            if not str(ref_name):
                continue
            r_desc = patch_descriptors.get(str(ref_name))
            if r_desc is None:
                continue
            if q_desc.ndim != 2 or r_desc.ndim != 2 or q_desc.shape[1] != r_desc.shape[1]:
                continue
            scores[row_idx, cand_idx] = float(_patch_match_score(q_desc, r_desc, topk=int(topk)))
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


def save_patch_descriptor_bank(
    path: str | Path,
    patch_descriptors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping | None = None,
) -> None:
    """Save a sample-name keyed patch descriptor bank with shape N,P,C."""
    names = sorted(str(name) for name in patch_descriptors)
    tensors = []
    for name in names:
        patches = F.normalize(patch_descriptors[name].detach().float().cpu(), dim=1, eps=1.0e-6)
        if patches.ndim != 2:
            raise ValueError("all patch descriptors must have shape (P,C)")
        tensors.append(patches)
    if tensors:
        shape = tuple(tensors[0].shape)
        if any(tuple(tensor.shape) != shape for tensor in tensors):
            raise ValueError("all patch descriptor tensors must have the same shape")
        tensor = torch.stack(tensors, dim=0)
    else:
        tensor = torch.empty((0, 0, 0), dtype=torch.float32)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "names": names,
            "patch_descriptors": tensor,
            "metadata": dict(metadata or {}),
        },
        out_path,
    )


def load_patch_descriptor_bank(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a patch descriptor bank saved by :func:`save_patch_descriptor_bank`."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    names = [str(name) for name in payload.get("names", [])]
    tensor = payload.get("patch_descriptors")
    if tensor is None:
        raise ValueError("patch descriptor bank missing 'patch_descriptors'")
    tensor = tensor.float().cpu()
    if tensor.ndim != 3 or tensor.shape[0] != len(names):
        raise ValueError("patch descriptor bank shape does not match names")
    descriptors = {
        name: F.normalize(tensor[idx].float(), dim=1, eps=1.0e-6)
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
