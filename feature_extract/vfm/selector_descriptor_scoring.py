"""Score fixed candidates using selector-projected descriptor banks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.score_table import ScoreRow
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("selector checkpoint must be a state_dict or contain one")
    if "projection.weight" in checkpoint:
        state_dict = checkpoint
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError("selector checkpoint does not contain projection.weight")
    if not isinstance(state_dict, Mapping):
        raise ValueError("selector state_dict must be a mapping")
    normalized = {}
    for key, value in state_dict.items():
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        normalized[name] = value
    return normalized


def infer_selector_dims_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    group_size: Optional[int] = None,
) -> tuple[int, int, int]:
    """Infer selector dimensions from a LocalizableFeatureSelector state_dict."""

    if "projection.weight" not in state_dict:
        raise ValueError("selector state_dict is missing projection.weight")
    projection_weight = state_dict["projection.weight"]
    if tuple(projection_weight.shape[-2:]) != (1, 1) or projection_weight.ndim != 4:
        raise ValueError("projection.weight must have shape (output_dim, input_dim, 1, 1)")
    inferred_output_dim = int(projection_weight.shape[0])
    inferred_input_dim = int(projection_weight.shape[1])
    resolved_input_dim = int(input_dim) if input_dim is not None else inferred_input_dim
    resolved_output_dim = int(output_dim) if output_dim is not None else inferred_output_dim

    if group_size is not None:
        resolved_group_size = int(group_size)
    else:
        if "group_logits" not in state_dict:
            raise ValueError("selector state_dict is missing group_logits; pass --group_size")
        group_count = int(state_dict["group_logits"].numel())
        if group_count <= 0 or resolved_input_dim % group_count != 0:
            raise ValueError("cannot infer selector group_size from group_logits")
        resolved_group_size = resolved_input_dim // group_count
    return resolved_input_dim, resolved_output_dim, resolved_group_size


def load_selector_from_checkpoint(
    checkpoint_path: Path,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    group_size: Optional[int] = None,
    device: str = "cpu",
) -> LocalizableFeatureSelector:
    """Load a LocalizableFeatureSelector from a state_dict checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    resolved_input_dim, resolved_output_dim, resolved_group_size = infer_selector_dims_from_state_dict(
        state_dict,
        input_dim=input_dim,
        output_dim=output_dim,
        group_size=group_size,
    )
    selector = LocalizableFeatureSelector(
        input_dim=resolved_input_dim,
        output_dim=resolved_output_dim,
        group_size=resolved_group_size,
    )
    selector.load_state_dict(state_dict)
    selector.to(torch.device(device))
    selector.eval()
    return selector


def _project_descriptor_bank(
    descriptors: TokenDescriptorBank,
    selector: LocalizableFeatureSelector,
    device: torch.device,
) -> np.ndarray:
    if descriptors.descriptors.shape[1] != selector.input_dim:
        raise ValueError(
            f"descriptor dimension {descriptors.descriptors.shape[1]} does not match selector input_dim {selector.input_dim}"
        )
    with torch.no_grad():
        tensor = torch.as_tensor(descriptors.descriptors, dtype=torch.float32, device=device)
        selected = selector(tensor.reshape(tensor.shape[0], tensor.shape[1], 1, 1)).selected.flatten(1)
        selected = F.normalize(selected, p=2, dim=1, eps=1e-6)
    return selected.detach().cpu().numpy().astype(np.float32, copy=False)


def score_candidate_bank_by_selector_descriptor_cosine(
    bank: CandidateHypothesisBank,
    query_descriptors: TokenDescriptorBank,
    map_descriptors: TokenDescriptorBank,
    selector: LocalizableFeatureSelector,
    method: str,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    device: str = "cpu",
) -> list[ScoreRow]:
    """Apply one selector to query/map descriptors, then score candidates by cosine."""

    if query_descriptors.layer_name != map_descriptors.layer_name:
        raise ValueError("query and map descriptor banks must use the same layer")
    if query_descriptors.descriptors.shape[1] != map_descriptors.descriptors.shape[1]:
        raise ValueError("query and map descriptor dimensions must match")

    torch_device = torch.device(device)
    selector = selector.to(torch_device)
    selector.eval()
    query_selected = _project_descriptor_bank(query_descriptors, selector, torch_device)
    map_selected = _project_descriptor_bank(map_descriptors, selector, torch_device)
    query_index = query_descriptors.index()
    map_index = map_descriptors.index()

    rows: list[ScoreRow] = []
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.reference_image is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing reference_image")
        if candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        if candidate.query_id not in query_index:
            raise ValueError(f"query descriptor not found: {candidate.query_id}")
        if candidate.reference_image not in map_index:
            raise ValueError(f"reference descriptor not found: {candidate.reference_image}")
        query_feature = query_selected[query_index[candidate.query_id]]
        map_feature = map_selected[map_index[candidate.reference_image]]
        rows.append(
            ScoreRow(
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                score=float(np.dot(query_feature, map_feature)),
                cost_m=float(candidate.pose_error.translation_m),
                basin_label=candidate.basin_label(
                    translation_threshold_m=translation_threshold_m,
                    rotation_threshold_deg=rotation_threshold_deg,
                ),
                protocol_kind=bank.protocol_kind,
                method=method,
            )
        )
    return rows
