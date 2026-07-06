from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from feature_extract.vfm.measurement_v1.rgb_patch_coarse_to_fine import (
    CoarseToFineRGBPatchMeasurementBranch,
    CoarseToFineRGBPatchMeasurementPrediction,
)
from feature_extract.vfm.measurement_v1.rgb_patch_coarse_to_fine_fusion import (
    apply_coarse_to_fine_rgb_patch_measurements_to_rows,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import RGBPatchMeasurementPrediction


def _write_query_image(path: Path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = np.arange(16, dtype=np.uint8).reshape(1, 16) * 8
    Image.fromarray(image, mode="RGB").save(path)


def _write_render_cache(path: Path) -> None:
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    rgb[..., 0] = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(1, 16)
    np.savez(path, rgb=rgb, depth=np.ones((16, 16), dtype=np.float32))


def test_coarse_to_fine_fusion_writes_final_measurement_without_moving_render_geometry(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png")
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache)
    model = CoarseToFineRGBPatchMeasurementBranch(
        coarse_search_radius_px=4.0,
        coarse_context_radius_px=1.0,
        coarse_step_px=2.0,
        fine_search_radius_px=1.0,
        fine_context_radius_px=1.0,
        fine_step_px=1.0,
        feature_dim=4,
        hidden_dim=8,
    )

    def fake_coarse_forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
        del render_patch, prior_scale_px
        batch = int(query_patch.shape[0])
        return RGBPatchMeasurementPrediction(
            logits=torch.zeros((batch, 1), dtype=torch.float32),
            offsets_xy=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            dustbin_logit=torch.full((batch,), -8.0, dtype=torch.float32),
        )

    def fake_fine_forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
        del render_patch, prior_scale_px
        batch = int(query_patch.shape[0])
        return RGBPatchMeasurementPrediction(
            logits=torch.zeros((batch, 1), dtype=torch.float32),
            offsets_xy=torch.tensor([[0.5, -0.5]], dtype=torch.float32),
            dustbin_logit=torch.full((batch,), -8.0, dtype=torch.float32),
        )

    model.coarse.forward_from_patches = types.MethodType(fake_coarse_forward_from_patches, model.coarse)
    model.fine.forward_from_patches = types.MethodType(fake_fine_forward_from_patches, model.fine)
    rows = [
        {
            "query_id": "q0.png",
            "query_center_x": "8.0",
            "query_center_y": "8.0",
            "render_x": "7.0",
            "render_y": "6.0",
            "render_depth": "5.0",
            "world_x": "1.0",
            "world_y": "2.0",
            "world_z": "3.0",
            "query_gt_x": "10.5",
            "query_gt_y": "7.5",
        }
    ]

    fused, summary = apply_coarse_to_fine_rgb_patch_measurements_to_rows(
        rows,
        image_root=image_root,
        render_cache_by_query={"q0.png": render_cache},
        model=model,
        query_image_width=16,
        query_image_height=16,
        render_image_width=16,
        render_image_height=16,
        batch_size=1,
        device=torch.device("cpu"),
    )

    row = fused[0]
    assert row["render_x"] == "7.0"
    assert row["render_depth"] == "5.0"
    assert row["world_x"] == "1.0"
    assert float(row["coarse_measurement_dx"]) == 2.0
    assert float(row["fine_measurement_dx"]) == 0.5
    assert float(row["measurement_dx"]) == 2.5
    assert float(row["query_refined_x"]) == 10.5
    assert float(row["query_refined_y"]) == 7.5
    assert summary["measurement_epe_median_px"] == 0.0
    assert summary["stage"] == "measurement_v1_rgb_patch_coarse_to_fine_match_table_fusion"


def test_coarse_to_fine_fusion_passes_prior_scale_to_coarse_stage(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _write_query_image(image_root / "q0.png")
    render_cache = tmp_path / "render_q0.npz"
    _write_render_cache(render_cache)
    model = CoarseToFineRGBPatchMeasurementBranch(
        coarse_search_radius_px=4.0,
        coarse_context_radius_px=1.0,
        coarse_step_px=2.0,
        fine_search_radius_px=1.0,
        fine_context_radius_px=1.0,
        fine_step_px=1.0,
        feature_dim=4,
        hidden_dim=8,
    )
    captured: dict[str, torch.Tensor | None] = {}

    def fake_coarse_forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
        del render_patch
        captured["coarse_prior_scale_px"] = prior_scale_px.detach().cpu() if prior_scale_px is not None else None
        batch = int(query_patch.shape[0])
        return RGBPatchMeasurementPrediction(
            logits=torch.zeros((batch, 1), dtype=torch.float32),
            offsets_xy=torch.zeros((1, 2), dtype=torch.float32),
            dustbin_logit=torch.full((batch,), -8.0, dtype=torch.float32),
        )

    def fake_fine_forward_from_patches(self, query_patch, render_patch, prior_scale_px=None):
        del render_patch, prior_scale_px
        batch = int(query_patch.shape[0])
        return RGBPatchMeasurementPrediction(
            logits=torch.zeros((batch, 1), dtype=torch.float32),
            offsets_xy=torch.zeros((1, 2), dtype=torch.float32),
            dustbin_logit=torch.full((batch,), -8.0, dtype=torch.float32),
        )

    model.coarse.forward_from_patches = types.MethodType(fake_coarse_forward_from_patches, model.coarse)
    model.fine.forward_from_patches = types.MethodType(fake_fine_forward_from_patches, model.fine)
    apply_coarse_to_fine_rgb_patch_measurements_to_rows(
        [
            {
                "query_id": "q0.png",
                "query_center_x": "8.0",
                "query_center_y": "8.0",
                "render_x": "7.0",
                "render_y": "6.0",
                "query_gt_x": "8.0",
                "query_gt_y": "8.0",
                "requested_residual_px": "12.5",
            }
        ],
        image_root=image_root,
        render_cache_by_query={"q0.png": render_cache},
        model=model,
        query_image_width=16,
        query_image_height=16,
        render_image_width=16,
        render_image_height=16,
        batch_size=1,
        device=torch.device("cpu"),
        coarse_prior_scale_key="requested_residual_px",
    )

    assert captured["coarse_prior_scale_px"] is not None
    assert torch.allclose(captured["coarse_prior_scale_px"], torch.tensor([12.5]))
