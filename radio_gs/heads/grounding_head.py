"""Text grounding head for RADIO-GS via SigLIP2-aligned similarity."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


class GroundingHead(nn.Module):
    """Computes text-to-feature similarity heatmaps from RADIO features.

    When ``use_adaptor=True``, a learned linear projection maps RADIO features
    into SigLIP2's embedding space before computing cosine similarity.

    Args:
        feature_dim: RADIO feature dimension (default 1280).
        adaptor_dim: Target dimension matching SigLIP2 text space (default 1152).
        use_adaptor: Whether to project visual features before comparison.
        temperature: Softmax temperature for cosine similarity.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        adaptor_dim: int = 1152,
        use_adaptor: bool = True,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_adaptor = use_adaptor

        if use_adaptor:
            self.adaptor = nn.Conv2d(feature_dim, adaptor_dim, 1, bias=False)
            self._visual_dim = adaptor_dim
        else:
            self.adaptor = None
            self._visual_dim = feature_dim

    def _project_features(self, features: Tensor) -> Tensor:
        if self.adaptor is not None:
            features = self.adaptor(features)
        return F.normalize(features, dim=1)

    @staticmethod
    def _normalize_text(text_embeddings: Tensor) -> Tensor:
        return F.normalize(text_embeddings, dim=-1)

    def forward(self, features: Tensor, text_embeddings: Tensor) -> Tensor:
        """Compute per-pixel similarity heatmaps for each text query.

        Args:
            features: [B, 1280, H, W] decoded RADIO visual features.
            text_embeddings: [B, N_queries, D_text] or [N_queries, D_text].
                D_text should match adaptor_dim (if use_adaptor) or feature_dim.

        Returns:
            [B, N_queries, H, W] cosine similarity heatmaps scaled by temperature.
        """
        B, _, H, W = features.shape
        vis = self._project_features(features)  # [B, D_vis, H, W]
        txt = self._normalize_text(text_embeddings)

        if txt.ndim == 2:
            txt = txt.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]

        # Project text to visual dim if sizes mismatch
        D_vis = self._visual_dim
        D_txt = txt.shape[-1]
        if D_txt != D_vis:
            if not hasattr(self, '_text_proj') or self._text_proj.in_features != D_txt:
                self._text_proj = nn.Linear(D_txt, D_vis, bias=False).to(txt.device)
            txt = F.normalize(self._text_proj(txt), dim=-1)

        vis_flat = vis.view(B, D_vis, H * W)  # [B, D, HW]
        sim = torch.bmm(txt, vis_flat)  # [B, N, HW]
        sim = sim.view(B, txt.shape[1], H, W)
        return sim / self.temperature

    def ground_text(
        self, features: Tensor, text_embeddings: Tensor, threshold: float = 0.5,
    ) -> Tensor:
        """Produce binary grounding masks by thresholding similarity.

        Args:
            features: [B, 1280, H, W] decoded RADIO visual features.
            text_embeddings: [B, N_queries, D_text] or [N_queries, D_text].
            threshold: Similarity threshold for binarisation.

        Returns:
            [B, N_queries, H, W] boolean masks.
        """
        sim = self.forward(features, text_embeddings)
        return sim > threshold


class GroundingLoss(nn.Module):
    """Loss for text-grounding supervision.

    Args:
        loss_type: One of 'bce', 'contrastive'.
        temperature: Temperature for the contrastive variant.
    """

    def __init__(self, loss_type: str = "bce", temperature: float = 0.07) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(self, pred_similarity: Tensor, gt_masks: Tensor) -> Tensor:
        """Compute grounding loss.

        Args:
            pred_similarity: [B, N_queries, H, W] predicted similarity heatmaps.
            gt_masks: [B, N_queries, H, W] binary ground-truth masks (float).

        Returns:
            Scalar loss tensor.
        """
        if self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(pred_similarity, gt_masks.float())
        elif self.loss_type == "contrastive":
            return self._contrastive_loss(pred_similarity, gt_masks)
        raise ValueError(f"Unknown loss_type '{self.loss_type}'")

    def _contrastive_loss(self, similarity: Tensor, gt_masks: Tensor) -> Tensor:
        """InfoNCE-style loss treating positive pixels as positives."""
        B, N, H, W = similarity.shape
        sim_flat = similarity.view(B * N, H * W) / self.temperature
        gt_flat = gt_masks.view(B * N, H * W).float()

        pos_mask = gt_flat > 0.5
        neg_mask = ~pos_mask

        losses = []
        for i in range(B * N):
            if pos_mask[i].sum() == 0 or neg_mask[i].sum() == 0:
                continue
            pos_logits = sim_flat[i][pos_mask[i]]
            all_logits = sim_flat[i]
            log_sum_exp = torch.logsumexp(all_logits, dim=0)
            loss_i = -(pos_logits - log_sum_exp).mean()
            losses.append(loss_i)

        if not losses:
            return similarity.sum() * 0.0
        return torch.stack(losses).mean()


def build_query_target_map(
    semantic_labels: Tensor,
    query_class_ids: List[int],
    ignore_index: int = -100,
) -> Tensor:
    """Map sparse semantic IDs to compact query indices for grounding CE loss."""
    if semantic_labels.ndim != 3:
        raise ValueError(
            f"Expected semantic_labels [B, H, W], got {tuple(semantic_labels.shape)}"
        )
    target = torch.full_like(semantic_labels, fill_value=ignore_index)
    for query_idx, class_id in enumerate(query_class_ids):
        target = torch.where(
            semantic_labels == int(class_id),
            torch.full_like(target, query_idx),
            target,
        )
    return target


class QueryGroundingAuxLoss(nn.Module):
    """Cross-entropy grounding loss over SigLIP-style text queries.

    This auxiliary loss is designed for training-time language calibration:
    projected visual features are compared against a fixed text bank, and only
    pixels belonging to grounding-eligible semantic classes contribute.
    """

    def __init__(
        self,
        feature_dim: int = 1536,
        temperature: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.grounding_head = GroundingHead(
            feature_dim=feature_dim,
            adaptor_dim=feature_dim,
            use_adaptor=False,
            temperature=temperature,
        )

    def forward(
        self,
        projected_features: Tensor,
        text_embeddings: Tensor,
        semantic_labels: Tensor,
        query_class_ids: List[int],
    ) -> dict[str, Tensor]:
        if projected_features.ndim != 4:
            raise ValueError(
                "Expected projected_features [B, C, H, W], "
                f"got {tuple(projected_features.shape)}"
            )
        if text_embeddings.ndim != 2:
            raise ValueError(
                f"Expected text_embeddings [N, C], got {tuple(text_embeddings.shape)}"
            )
        if semantic_labels.ndim != 3:
            raise ValueError(
                f"Expected semantic_labels [B, H, W], got {tuple(semantic_labels.shape)}"
            )
        if projected_features.shape[0] != semantic_labels.shape[0]:
            raise ValueError(
                "Batch size mismatch between projected_features and semantic_labels: "
                f"{projected_features.shape[0]} vs {semantic_labels.shape[0]}"
            )
        if projected_features.shape[-2:] != semantic_labels.shape[-2:]:
            raise ValueError(
                "Spatial size mismatch between projected_features and semantic_labels: "
                f"{tuple(projected_features.shape[-2:])} vs {tuple(semantic_labels.shape[-2:])}"
            )
        if projected_features.shape[1] != text_embeddings.shape[1]:
            raise ValueError(
                "Feature/text dim mismatch for QueryGroundingAuxLoss: "
                f"{projected_features.shape[1]} vs {text_embeddings.shape[1]}"
            )

        targets = build_query_target_map(
            semantic_labels,
            query_class_ids,
            ignore_index=self.ignore_index,
        )
        valid_mask = targets != self.ignore_index
        zero = projected_features.sum() * 0.0
        if not valid_mask.any():
            return {
                "loss": zero,
                "accuracy": zero.detach(),
                "valid_ratio": zero.detach(),
            }

        logits = self.grounding_head(projected_features, text_embeddings)
        loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            accuracy = (pred[valid_mask] == targets[valid_mask]).float().mean()
            valid_ratio = valid_mask.float().mean()
        return {
            "loss": loss,
            "accuracy": accuracy,
            "valid_ratio": valid_ratio,
        }


def compute_grounding_iou(
    pred_mask: Tensor, gt_mask: Tensor, threshold: float = 0.5,
) -> float:
    """Compute IoU between predicted and ground-truth grounding masks.

    Args:
        pred_mask: [B, N, H, W] predicted similarity or binary mask.
        gt_mask: [B, N, H, W] binary ground-truth mask.
        threshold: Threshold applied to ``pred_mask`` if not already binary.

    Returns:
        Mean IoU as a Python float.
    """
    pred_bin = (pred_mask > threshold).bool()
    gt_bin = gt_mask.bool()
    intersection = (pred_bin & gt_bin).float().sum()
    union = (pred_bin | gt_bin).float().sum()
    if union == 0:
        return 0.0
    return (intersection / union).item()


def compute_grounding_ap(pred_similarity: Tensor, gt_mask: Tensor) -> float:
    """Compute average precision for grounding predictions.

    Args:
        pred_similarity: [B, N, H, W] predicted similarity heatmap.
        gt_mask: [B, N, H, W] binary ground-truth mask.

    Returns:
        Average precision as a Python float.
    """
    scores = pred_similarity.detach().flatten()
    labels = gt_mask.detach().flatten().bool()

    if labels.sum() == 0 or (~labels).sum() == 0:
        return 0.0

    sorted_idx = scores.argsort(descending=True)
    labels_sorted = labels[sorted_idx].float()

    tp_cumsum = labels_sorted.cumsum(dim=0)
    precision = tp_cumsum / torch.arange(1, len(labels_sorted) + 1, device=scores.device, dtype=scores.dtype)
    recall_change = labels_sorted
    ap = (precision * recall_change).sum() / labels_sorted.sum()
    return ap.item()
