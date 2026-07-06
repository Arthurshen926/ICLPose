from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    RGBPatchMeasurementBranch,
    RGBPatchMeasurementPrediction,
    crop_rgb_window,
)


@dataclass(frozen=True)
class CoarseToFineRGBPatchMeasurementPrediction:
    coarse: RGBPatchMeasurementPrediction
    fine: RGBPatchMeasurementPrediction
    coarse_offset_xy: torch.Tensor
    coarse_recenter_offset_xy: torch.Tensor
    fine_offset_xy: torch.Tensor
    final_offset_xy: torch.Tensor
    coarse_cov_2x2: torch.Tensor
    fine_cov_2x2: torch.Tensor
    final_cov_2x2: torch.Tensor
    dustbin_logit: torch.Tensor
    valid_probability: torch.Tensor


def likelihood_moments(
    logits: torch.Tensor,
    offsets_xy: torch.Tensor,
    *,
    covariance_floor_px2: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=1)
    offsets = offsets_xy.to(device=logits.device, dtype=logits.dtype).reshape(1, -1, 2)
    mean = torch.sum(probs[..., None] * offsets, dim=1)
    centered = offsets - mean[:, None, :]
    cov = torch.einsum("bk,bki,bkj->bij", probs, centered, centered)
    eye = torch.eye(2, device=logits.device, dtype=logits.dtype).reshape(1, 2, 2)
    return mean, cov + eye * float(covariance_floor_px2)


def likelihood_mode(logits: torch.Tensor, offsets_xy: torch.Tensor) -> torch.Tensor:
    offsets = offsets_xy.to(device=logits.device, dtype=logits.dtype)
    peak_index = torch.argmax(logits, dim=1)
    return offsets[peak_index]


class CoarseToFineRGBPatchMeasurementBranch(nn.Module):
    """Two-stage template-to-search branch.

    The coarse branch covers a large residual basin. The fine branch is cropped
    around the coarse likelihood mean and predicts only the remaining residual.
    Render-side geometry stays fixed; only query-side measurement is refined.
    """

    def __init__(
        self,
        *,
        coarse_search_radius_px: float,
        coarse_context_radius_px: float,
        coarse_step_px: float,
        fine_search_radius_px: float,
        fine_context_radius_px: float,
        fine_step_px: float,
        feature_dim: int = 32,
        hidden_dim: int | None = None,
        input_mode: str = "rgb",
        coarse_template_scale_factors: Sequence[float] = (1.0,),
        fine_template_scale_factors: Sequence[float] = (1.0,),
        coarse_recenter_head: str = "mode",
    ) -> None:
        super().__init__()
        recenter = str(coarse_recenter_head)
        if recenter not in {"likelihood", "mode"}:
            raise ValueError("coarse_recenter_head must be 'likelihood' or 'mode'")
        self.coarse_recenter_head = recenter
        self.coarse = RGBPatchMeasurementBranch(
            search_radius_px=float(coarse_search_radius_px),
            context_radius_px=float(coarse_context_radius_px),
            step_px=float(coarse_step_px),
            feature_dim=int(feature_dim),
            hidden_dim=hidden_dim,
            input_mode=str(input_mode),
            template_scale_factors=tuple(float(value) for value in coarse_template_scale_factors),
        )
        self.fine = RGBPatchMeasurementBranch(
            search_radius_px=float(fine_search_radius_px),
            context_radius_px=float(fine_context_radius_px),
            step_px=float(fine_step_px),
            feature_dim=int(feature_dim),
            hidden_dim=hidden_dim,
            input_mode=str(input_mode),
            template_scale_factors=tuple(float(value) for value in fine_template_scale_factors),
        )

    @property
    def coarse_crop_radius_px(self) -> float:
        return float(self.coarse.crop_radius_px)

    @property
    def fine_crop_radius_px(self) -> float:
        return float(self.fine.crop_radius_px)

    def _crop_query_patch(
        self,
        images: torch.Tensor,
        centers_xy: torch.Tensor,
        *,
        radius_px: float,
        step_px: float,
        image_width: int,
        image_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return crop_rgb_window(
            images,
            centers_xy,
            radius_px=float(radius_px),
            step_px=float(step_px),
            image_width=int(image_width),
            image_height=int(image_height),
        )

    def _crop_render_patch(
        self,
        images: torch.Tensor,
        centers_xy: torch.Tensor,
        *,
        radius_px: float,
        step_px: float,
        image_width: int,
        image_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return crop_rgb_window(
            images,
            centers_xy,
            radius_px=float(radius_px),
            step_px=float(step_px),
            image_width=int(image_width),
            image_height=int(image_height),
        )

    def forward_from_images(
        self,
        *,
        query_images: torch.Tensor,
        render_images: torch.Tensor,
        query_centers_xy: torch.Tensor,
        render_anchor_xy: torch.Tensor,
        query_image_width: int,
        query_image_height: int,
        render_image_width: int,
        render_image_height: int,
        coarse_prior_scale_px: torch.Tensor | None = None,
        fine_prior_scale_px: torch.Tensor | None = None,
        detach_coarse_for_fine_crop: bool = True,
    ) -> CoarseToFineRGBPatchMeasurementPrediction:
        coarse_query_patch, _ = self._crop_query_patch(
            query_images,
            query_centers_xy,
            radius_px=self.coarse.crop_radius_px,
            step_px=self.coarse.step_px,
            image_width=int(query_image_width),
            image_height=int(query_image_height),
        )
        coarse_render_patch, _ = self._crop_render_patch(
            render_images,
            render_anchor_xy,
            radius_px=self.coarse.crop_radius_px,
            step_px=self.coarse.step_px,
            image_width=int(render_image_width),
            image_height=int(render_image_height),
        )
        coarse_pred = self.coarse.forward_from_patches(
            coarse_query_patch,
            coarse_render_patch,
            prior_scale_px=coarse_prior_scale_px,
        )
        coarse_mean, coarse_cov = likelihood_moments(coarse_pred.logits, coarse_pred.offsets_xy)
        if self.coarse_recenter_head == "mode":
            recenter_offset = likelihood_mode(coarse_pred.logits, coarse_pred.offsets_xy)
        else:
            recenter_offset = coarse_mean
        crop_offset = recenter_offset.detach() if bool(detach_coarse_for_fine_crop) else recenter_offset
        fine_centers = query_centers_xy.to(device=query_images.device, dtype=query_images.dtype).reshape(int(query_images.shape[0]), 2) + crop_offset
        fine_query_patch, _ = self._crop_query_patch(
            query_images,
            fine_centers,
            radius_px=self.fine.crop_radius_px,
            step_px=self.fine.step_px,
            image_width=int(query_image_width),
            image_height=int(query_image_height),
        )
        fine_render_patch, _ = self._crop_render_patch(
            render_images,
            render_anchor_xy,
            radius_px=self.fine.crop_radius_px,
            step_px=self.fine.step_px,
            image_width=int(render_image_width),
            image_height=int(render_image_height),
        )
        fine_pred = self.fine.forward_from_patches(
            fine_query_patch,
            fine_render_patch,
            prior_scale_px=fine_prior_scale_px,
        )
        fine_mean, fine_cov = likelihood_moments(fine_pred.logits, fine_pred.offsets_xy)
        final_offset = recenter_offset + fine_mean
        final_cov = coarse_cov + fine_cov
        coarse_dustbin = torch.sigmoid(coarse_pred.dustbin_logit.reshape(-1))
        fine_dustbin = torch.sigmoid(fine_pred.dustbin_logit.reshape(-1))
        valid_probability = (1.0 - coarse_dustbin) * (1.0 - fine_dustbin)
        dustbin_prob = (1.0 - valid_probability).clamp(1e-6, 1.0 - 1e-6)
        dustbin_logit = torch.logit(dustbin_prob)
        return CoarseToFineRGBPatchMeasurementPrediction(
            coarse=coarse_pred,
            fine=fine_pred,
            coarse_offset_xy=coarse_mean,
            coarse_recenter_offset_xy=recenter_offset,
            fine_offset_xy=fine_mean,
            final_offset_xy=final_offset,
            coarse_cov_2x2=coarse_cov,
            fine_cov_2x2=fine_cov,
            final_cov_2x2=final_cov,
            dustbin_logit=dustbin_logit,
            valid_probability=valid_probability,
        )


def load_coarse_to_fine_rgb_patch_measurement_branch(
    *,
    coarse_checkpoint: Path,
    fine_checkpoint: Path,
    device: torch.device,
) -> CoarseToFineRGBPatchMeasurementBranch:
    from feature_extract.vfm.measurement_v1.rgb_patch_match_table_fusion import load_rgb_patch_measurement_branch

    coarse = load_rgb_patch_measurement_branch(Path(coarse_checkpoint), device=device)
    fine = load_rgb_patch_measurement_branch(Path(fine_checkpoint), device=device)
    model = CoarseToFineRGBPatchMeasurementBranch(
        coarse_search_radius_px=coarse.search_radius_px,
        coarse_context_radius_px=coarse.context_radius_px,
        coarse_step_px=coarse.step_px,
        fine_search_radius_px=fine.search_radius_px,
        fine_context_radius_px=fine.context_radius_px,
        fine_step_px=fine.step_px,
        feature_dim=1,
        coarse_recenter_head="mode",
    )
    model.coarse = coarse
    model.fine = fine
    return model.to(device).eval()
