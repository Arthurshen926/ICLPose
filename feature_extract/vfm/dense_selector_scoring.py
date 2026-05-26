"""Score fixed candidates with dense-token selected descriptors."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.score_table import ScoreRow
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord


def _validated_record_index(manifest: TokenBankManifest) -> dict[str, TokenBankRecord]:
    manifest.validate(verify_checksums=False)
    return {record.image_id: record for record in manifest.records}


def _load_layer(record: TokenBankRecord, layer_name: str) -> np.ndarray:
    with np.load(record.token_path) as data:
        if layer_name not in data:
            raise ValueError(f"layer {layer_name!r} not found in {record.token_path}")
        tokens = np.asarray(data[layer_name], dtype=np.float32)
    if tokens.ndim != 3:
        raise ValueError("dense token feature map must have shape (C, H, W)")
    return tokens


def dense_selected_descriptor(
    records: Mapping[str, TokenBankRecord],
    image_id: str,
    selector: LocalizableFeatureSelector,
    layer_name: str,
    device: str = "cpu",
    utility_weighted_pooling: bool = False,
) -> np.ndarray:
    """Run one selector over a dense token map and mean-pool selected features."""

    if image_id not in records:
        raise ValueError(f"token record not found: {image_id}")
    tokens = _load_layer(records[image_id], layer_name)
    if tokens.shape[0] != selector.input_dim:
        raise ValueError(f"token channels {tokens.shape[0]} do not match selector input_dim {selector.input_dim}")

    torch_device = torch.device(device)
    selector = selector.to(torch_device)
    selector.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(tokens[None, ...], dtype=torch.float32, device=torch_device)
        output = selector(tensor)
        selected = output.selected
        if utility_weighted_pooling:
            weights = output.utility.clamp_min(1e-6)
            descriptor = (selected * weights).flatten(2).sum(dim=2)
            descriptor = descriptor / weights.flatten(2).sum(dim=2).clamp_min(1e-6)
        else:
            descriptor = selected.flatten(2).mean(dim=2)
        descriptor = F.normalize(descriptor, p=2, dim=1, eps=1e-6)
    return descriptor[0].detach().cpu().numpy().astype(np.float32, copy=False)


def score_candidate_bank_by_dense_selector(
    bank: CandidateHypothesisBank,
    query_manifest: TokenBankManifest,
    map_manifest: TokenBankManifest,
    selector: LocalizableFeatureSelector,
    layer_name: str,
    method: str,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    device: str = "cpu",
    utility_weighted_pooling: bool = False,
) -> list[ScoreRow]:
    """Score candidates by cosine similarity of dense selected descriptors."""

    query_index = _validated_record_index(query_manifest)
    map_index = _validated_record_index(map_manifest)
    query_cache: dict[str, np.ndarray] = {}
    map_cache: dict[str, np.ndarray] = {}
    rows: list[ScoreRow] = []
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.reference_image is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing reference_image")
        if candidate.pose_error is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose_error")
        if candidate.query_id not in query_cache:
            query_cache[candidate.query_id] = dense_selected_descriptor(
                query_index,
                candidate.query_id,
                selector,
                layer_name,
                device=device,
                utility_weighted_pooling=utility_weighted_pooling,
            )
        if candidate.reference_image not in map_cache:
            map_cache[candidate.reference_image] = dense_selected_descriptor(
                map_index,
                candidate.reference_image,
                selector,
                layer_name,
                device=device,
                utility_weighted_pooling=utility_weighted_pooling,
            )
        rows.append(
            ScoreRow(
                query_id=candidate.query_id,
                candidate_id=candidate.candidate_id,
                score=float(np.dot(query_cache[candidate.query_id], map_cache[candidate.reference_image])),
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
