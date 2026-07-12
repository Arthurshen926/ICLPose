"""Feature mapper interfaces for raw RADIO descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from feature_extract.vfm.localization.schemas import MappedFeatureMap
from feature_extract.vfm.matcha_coarse_fine_adapter import (
    MatchaCoarseFineAdapter,
    project_feature_map_with_matcha_adapter,
)
from feature_extract.vfm.matcha_joint_training import (
    MatchaStyleJointModel,
    RadioDualAttentionFusionJointModel,
)


class FeatureMapper(Protocol):
    """Map one raw RADIO feature map to descriptors used by localization."""

    def project(self, feature_map: np.ndarray) -> MappedFeatureMap:
        """Return coarse descriptors and optional measurement context."""


@dataclass
class AdapterFeatureMapper:
    """Compatibility mapper backed by the existing MATCHA coarse adapter."""

    model: MatchaCoarseFineAdapter
    device: str = "cpu"
    batch_size: int = 65536

    def project(self, feature_map: np.ndarray) -> MappedFeatureMap:
        descriptors, offset_logits = project_feature_map_with_matcha_adapter(
            self.model,
            feature_map,
            device=str(self.device),
            batch_size=int(self.batch_size),
        )
        return MappedFeatureMap(
            coarse_descriptors=descriptors,
            measurement_context=descriptors,
            offset_logits=offset_logits,
            heatmap=None,
        )


@dataclass
class JointFeatureMapper:
    """Feature mapper backed by a full MATCHA joint model checkpoint."""

    model: MatchaStyleJointModel | RadioDualAttentionFusionJointModel
    device: str = "cpu"

    def project(self, feature_map: np.ndarray) -> MappedFeatureMap:
        fmap = np.asarray(feature_map, dtype=np.float32)
        if fmap.ndim != 3:
            raise ValueError("feature_map must have shape (C, H, W)")
        return self.project_batch(fmap[None])[0]

    def project_batch(self, feature_maps: np.ndarray) -> list[MappedFeatureMap]:
        """Project a same-shape image batch in one mapper forward."""

        maps = np.asarray(feature_maps, dtype=np.float32)
        if maps.ndim != 4:
            raise ValueError("feature_maps must have shape (B, C, H, W)")
        torch_device = torch.device(self.device if torch.cuda.is_available() or not str(self.device).startswith("cuda") else "cpu")
        was_training = self.model.training
        model = self.model.to(torch_device).eval()
        tensor = torch.as_tensor(maps, dtype=torch.float32, device=torch_device)
        with torch.no_grad():
            if isinstance(model, RadioDualAttentionFusionJointModel) and str(model.attention_fusion_mode) == "matcha_original":
                coarse_desc, measurement_context, heatmap_logits = model.forward_fuse_feature(tensor)
                offset_logits = model.offset_head_map(measurement_context)
            else:
                coarse_desc, heatmap_logits, offset_logits = model.forward_feature_map(tensor)
                measurement_context = coarse_desc
        if was_training:
            self.model.train()
        coarse_np = coarse_desc.detach().cpu().numpy().astype(np.float32, copy=False)
        context_np = measurement_context.detach().cpu().numpy().astype(np.float32, copy=False)
        offsets_np = offset_logits.detach().cpu().numpy().astype(np.float32, copy=False)
        heatmap_np = torch.sigmoid(heatmap_logits[:, 0]).detach().cpu().numpy().astype(np.float32, copy=False)
        return [
            MappedFeatureMap(
                coarse_descriptors=coarse_np[index],
                measurement_context=context_np[index],
                offset_logits=offsets_np[index],
                heatmap=heatmap_np[index],
            )
            for index in range(int(maps.shape[0]))
        ]
