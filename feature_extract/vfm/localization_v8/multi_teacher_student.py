"""Multi-teacher/single-student adaptor for canonical localization features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)


@dataclass(frozen=True)
class MapletRetrievalAdaptorConfig:
    feature_dim: int = 128
    hidden_dim: int = 256
    residual_scale: float = 0.25


class MapletRetrievalAdaptor(nn.Module):
    """One localization readout; downstream teacher spaces are not retained."""

    def __init__(self, config: MapletRetrievalAdaptorConfig) -> None:
        super().__init__()
        self.config = config
        self.residual = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.feature_dim),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        base = F.normalize(feature, dim=-1)
        update = self.residual(base)
        return F.normalize(base + float(self.config.residual_scale) * update, dim=-1)

    def project_numpy(self, feature: np.ndarray, *, device: str = "cpu") -> np.ndarray:
        value = torch.from_numpy(np.asarray(feature, dtype=np.float32)).to(device)
        with torch.inference_mode():
            output = self(value).cpu().numpy()
        return output.astype(np.float32)


def save_maplet_retrieval_adaptor(
    model: MapletRetrievalAdaptor,
    path: Path,
    *,
    metadata: Mapping[str, object],
) -> None:
    contract = dict(metadata)
    contract.update(
        artifact_type="v8_multi_teacher_single_student_adaptor",
        stored_feature_type="canonical_radio_localization_feature",
        stores_teacher_embeddings=False,
        stores_mapping_rgb=False,
        stores_mapping_image_ids=False,
        stores_mapping_image_paths=False,
        uses_sfm_points=False,
        uses_sfm_tracks=False,
        uses_alike_descriptors=False,
        uses_radio_intermediate=False,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
            "metadata": contract,
        },
        path,
    )


def load_maplet_retrieval_adaptor(
    path: Path, *, device: str = "cpu"
) -> tuple[MapletRetrievalAdaptor, dict[str, object]]:
    payload = torch.load(path, map_location=device)
    metadata = dict(payload["metadata"])
    if metadata.get("artifact_type") != "v8_multi_teacher_single_student_adaptor":
        raise ValueError("not a V8 single-student adaptor")
    for key in ("stores_teacher_embeddings", "stores_mapping_rgb", "uses_sfm_points", "uses_sfm_tracks"):
        if bool(metadata.get(key, False)):
            raise ValueError(f"single-student adaptor violates contract: {key}")
    model = MapletRetrievalAdaptor(
        MapletRetrievalAdaptorConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


def transform_canonical_maplet_bank(
    bank: SurfaceRetrievalMapletBank,
    adaptor: MapletRetrievalAdaptor,
    *,
    adaptor_path: Path,
    device: str = "cpu",
) -> SurfaceRetrievalMapletBank:
    metadata = dict(bank.metadata or {})
    metadata.update(
        representation="single_multi_teacher_distilled_radio_localization_mixture",
        localization_adaptor_sha256=hashlib.sha256(Path(adaptor_path).read_bytes()).hexdigest(),
        stores_downstream_embeddings=False,
        teacher_embeddings_discarded_after_training=True,
    )
    return SurfaceRetrievalMapletBank(
        maplet_ids=bank.maplet_ids,
        centers=bank.centers,
        normals=bank.normals,
        extents=bank.extents,
        tangent_frames=bank.tangent_frames,
        descriptor_offsets=bank.descriptor_offsets,
        descriptors=adaptor.project_numpy(bank.descriptors, device=device),
        descriptor_weights=bank.descriptor_weights,
        descriptor_centers=bank.descriptor_centers,
        descriptor_covariances=bank.descriptor_covariances,
        quality_scores=bank.quality_scores,
        descriptor_uncertainties=bank.descriptor_uncertainties,
        query_projection=None,
        metadata=metadata,
    )
