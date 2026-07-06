from __future__ import annotations

import types

import torch

from feature_extract.vfm.measurement_v1.rgb_patch_coarse_to_fine import CoarseToFineRGBPatchMeasurementBranch
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import RGBPatchMeasurementPrediction, local_offset_grid


def _fake_prediction(*, search_radius_px: float, step_px: float, dx: float, dy: float) -> RGBPatchMeasurementPrediction:
    offsets = local_offset_grid(search_radius_px=search_radius_px, step_px=step_px)
    logits = torch.full((1, int(offsets.shape[0])), -80.0, dtype=torch.float32)
    target = torch.tensor([dx, dy], dtype=torch.float32)
    index = int(torch.argmin(torch.linalg.norm(offsets - target.reshape(1, 2), dim=1)).item())
    logits[0, index] = 80.0
    return RGBPatchMeasurementPrediction(
        logits=logits,
        offsets_xy=offsets,
        dustbin_logit=torch.tensor([-8.0], dtype=torch.float32),
    )


def _fake_multimodal_prediction(
    *,
    search_radius_px: float,
    step_px: float,
    primary_dx: float,
    primary_dy: float,
    secondary_dx: float,
    secondary_dy: float,
) -> RGBPatchMeasurementPrediction:
    offsets = local_offset_grid(search_radius_px=search_radius_px, step_px=step_px)
    logits = torch.full((1, int(offsets.shape[0])), -80.0, dtype=torch.float32)
    for dx, dy, logit in ((primary_dx, primary_dy, 4.0), (secondary_dx, secondary_dy, 3.9)):
        target = torch.tensor([dx, dy], dtype=torch.float32)
        index = int(torch.argmin(torch.linalg.norm(offsets - target.reshape(1, 2), dim=1)).item())
        logits[0, index] = logit
    return RGBPatchMeasurementPrediction(
        logits=logits,
        offsets_xy=offsets,
        dustbin_logit=torch.tensor([-8.0], dtype=torch.float32),
    )


def test_coarse_to_fine_prediction_composes_offsets_and_recenters_fine_search() -> None:
    model = CoarseToFineRGBPatchMeasurementBranch(
        coarse_search_radius_px=4.0,
        coarse_context_radius_px=1.0,
        coarse_step_px=2.0,
        fine_search_radius_px=1.0,
        fine_context_radius_px=1.0,
        fine_step_px=1.0,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
    )
    fine_centers: list[torch.Tensor] = []

    def coarse_forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
        return _fake_prediction(search_radius_px=4.0, step_px=2.0, dx=2.0, dy=0.0)

    def fine_forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
        return _fake_prediction(search_radius_px=1.0, step_px=1.0, dx=1.0, dy=0.0)

    original_crop = model._crop_query_patch

    def record_query_crop(images, centers_xy, **kwargs):
        fine_centers.append(centers_xy.detach().cpu().clone())
        return original_crop(images, centers_xy, **kwargs)

    model.coarse.forward_from_patches = types.MethodType(coarse_forward_from_patches, model.coarse)
    model.fine.forward_from_patches = types.MethodType(fine_forward_from_patches, model.fine)
    model._crop_query_patch = record_query_crop  # type: ignore[method-assign]

    query = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    render = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    pred = model.forward_from_images(
        query_images=query,
        render_images=render,
        query_centers_xy=torch.tensor([[8.0, 8.0]], dtype=torch.float32),
        render_anchor_xy=torch.tensor([[8.0, 8.0]], dtype=torch.float32),
        query_image_width=16,
        query_image_height=16,
        render_image_width=16,
        render_image_height=16,
    )

    assert torch.allclose(pred.coarse_offset_xy, torch.tensor([[2.0, 0.0]]), atol=1e-4)
    assert torch.allclose(pred.fine_offset_xy, torch.tensor([[1.0, 0.0]]), atol=1e-4)
    assert torch.allclose(pred.final_offset_xy, torch.tensor([[3.0, 0.0]]), atol=1e-4)
    assert len(fine_centers) == 2
    assert torch.allclose(fine_centers[0], torch.tensor([[8.0, 8.0]]), atol=1e-4)
    assert torch.allclose(fine_centers[1], torch.tensor([[10.0, 8.0]]), atol=1e-4)


def test_coarse_to_fine_can_recenter_fine_search_on_coarse_mode_when_posterior_is_multimodal() -> None:
    model = CoarseToFineRGBPatchMeasurementBranch(
        coarse_search_radius_px=4.0,
        coarse_context_radius_px=1.0,
        coarse_step_px=2.0,
        fine_search_radius_px=1.0,
        fine_context_radius_px=1.0,
        fine_step_px=1.0,
        feature_dim=4,
        hidden_dim=8,
        input_mode="rgb",
        coarse_recenter_head="mode",
    )
    fine_centers: list[torch.Tensor] = []

    def coarse_forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
        return _fake_multimodal_prediction(
            search_radius_px=4.0,
            step_px=2.0,
            primary_dx=4.0,
            primary_dy=0.0,
            secondary_dx=-4.0,
            secondary_dy=0.0,
        )

    def fine_forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
        return _fake_prediction(search_radius_px=1.0, step_px=1.0, dx=0.0, dy=0.0)

    original_crop = model._crop_query_patch

    def record_query_crop(images, centers_xy, **kwargs):
        fine_centers.append(centers_xy.detach().cpu().clone())
        return original_crop(images, centers_xy, **kwargs)

    model.coarse.forward_from_patches = types.MethodType(coarse_forward_from_patches, model.coarse)
    model.fine.forward_from_patches = types.MethodType(fine_forward_from_patches, model.fine)
    model._crop_query_patch = record_query_crop  # type: ignore[method-assign]

    query = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    render = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    pred = model.forward_from_images(
        query_images=query,
        render_images=render,
        query_centers_xy=torch.tensor([[8.0, 8.0]], dtype=torch.float32),
        render_anchor_xy=torch.tensor([[8.0, 8.0]], dtype=torch.float32),
        query_image_width=16,
        query_image_height=16,
        render_image_width=16,
        render_image_height=16,
    )

    assert len(fine_centers) == 2
    assert torch.allclose(fine_centers[0], torch.tensor([[8.0, 8.0]]), atol=1e-4)
    assert torch.allclose(fine_centers[1], torch.tensor([[12.0, 8.0]]), atol=1e-4)
    assert torch.allclose(pred.coarse_recenter_offset_xy, torch.tensor([[4.0, 0.0]]), atol=1e-4)
