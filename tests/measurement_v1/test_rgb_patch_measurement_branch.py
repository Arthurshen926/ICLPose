from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    RGBPatchMeasurementBranch,
    RGBPatchMeasurementPrediction,
    TexturePatchEncoder,
    coarse_to_fine_template_search_cost_volume_logits,
    continuous_offset_nll_with_dustbin,
    cost_volume_quality_features,
    crop_rgb_window,
    crop_rgb_window_with_source_from_output_affine,
    crop_rgb_window_with_source_from_output_homography,
    local_offset_grid,
    residual_delta_gaussian_nll,
    template_search_cost_volume_logits,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import train_rgb_patch_measurement_branch
from feature_extract.vfm.measurement_v1 import rgb_patch_training
from feature_extract.vfm.measurement_v1.rgb_patch_training import augment_render_template_patch


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_split_train_val_can_keep_query_groups_disjoint() -> None:
    rows = [
        {"query_id": "q0", "track_id": "1"},
        {"query_id": "q0", "track_id": "2"},
        {"query_id": "q1", "track_id": "3"},
        {"query_id": "q1", "track_id": "4"},
        {"query_id": "q2", "track_id": "5"},
        {"query_id": "q2", "track_id": "6"},
    ]

    train_rows, val_rows = rgb_patch_training._split_train_val(rows, val_fraction=0.34, seed=3, group_key="query_id")

    train_queries = {row["query_id"] for row in train_rows}
    val_queries = {row["query_id"] for row in val_rows}
    assert train_rows
    assert val_rows
    assert train_queries.isdisjoint(val_queries)
    assert train_queries | val_queries == {"q0", "q1", "q2"}


def test_residual_bin_metrics_report_per_requested_residual_performance() -> None:
    rows = [
        {"requested_residual_px": "1.0"},
        {"requested_residual_px": "1.0"},
        {"requested_residual_px": "2.0"},
    ]
    epe = torch.tensor([0.25, 0.75, 1.25], dtype=torch.float32)
    baseline = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)

    metrics = rgb_patch_training._residual_bin_metrics(rows, epe, baseline, prefix="likelihood")

    assert metrics["likelihood_bin_1p000_count"] == 2
    assert metrics["likelihood_bin_1p000_baseline_epe_median_px"] == 1.0
    assert metrics["likelihood_bin_1p000_epe_median_px"] == 0.25
    assert metrics["likelihood_bin_1p000_recall_0p5px"] == 0.5
    assert metrics["likelihood_bin_1p000_worsen_ratio"] == 0.0
    assert metrics["likelihood_bin_2p000_count"] == 1
    assert metrics["likelihood_bin_2p000_improve_ratio"] == 1.0


def test_categorical_group_metrics_report_policy_specific_performance() -> None:
    rows = [
        {"measurement_policy": "center_preserve"},
        {"measurement_policy": "teacher_correction"},
        {"measurement_policy": "teacher_correction"},
    ]
    epe = torch.tensor([0.1, 0.4, 1.2], dtype=torch.float32)
    baseline = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)

    metrics = rgb_patch_training._categorical_group_metrics(
        rows,
        epe,
        baseline,
        prefix="likelihood_policy",
        group_key="measurement_policy",
    )

    assert metrics["likelihood_policy_center_preserve_count"] == 1
    assert metrics["likelihood_policy_center_preserve_improve_ratio"] == 0.0
    assert metrics["likelihood_policy_teacher_correction_count"] == 2
    assert metrics["likelihood_policy_teacher_correction_epe_median_px"] == 0.4000000059604645
    assert metrics["likelihood_policy_teacher_correction_improve_ratio"] == 0.5


def test_crop_rgb_window_uses_original_pixel_coordinates_and_subpixel_step() -> None:
    image = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    image[:, 0] = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    image[:, 1] = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)

    patch, offsets = crop_rgb_window(
        image,
        torch.tensor([[4.0, 3.0]], dtype=torch.float32),
        radius_px=1.0,
        step_px=0.5,
        image_width=8,
        image_height=8,
    )

    assert patch.shape == (1, 3, 5, 5)
    assert offsets.shape == (25, 2)
    assert torch.allclose(patch[0, :2, 2, 2], torch.tensor([4.0, 3.0]), atol=1e-5)
    assert torch.allclose(offsets[12], torch.tensor([0.0, 0.0]))


def test_rgb_patch_stack_batch_can_cache_images_on_requested_device(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = np.arange(8, dtype=np.uint8).reshape(1, 8) * 16
    image[..., 1] = np.arange(8, dtype=np.uint8).reshape(8, 1) * 16
    Image.fromarray(image, mode="RGB").save(image_root / "q0.png")
    render_rgb = np.zeros((8, 8, 3), dtype=np.float32)
    render_rgb[..., 0] = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(1, 8)
    render_rgb[..., 1] = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(8, 1)
    render_cache_path = tmp_path / "render_q0.npz"
    np.savez(render_cache_path, rgb=render_rgb, depth=np.ones((8, 8), dtype=np.float32))
    rows = [
        {
            "query_id": "q0.png",
            "center_x": "4.0",
            "center_y": "4.0",
            "query_gt_x": "4.5",
            "query_gt_y": "4.0",
            "render_x": "4.0",
            "render_y": "4.0",
        }
    ]
    kwargs = dict(
        rows=rows,
        image_root=image_root,
        render_cache_by_query={"q0.png": render_cache_path},
        image_width=8,
        image_height=8,
        crop_radius_px=1.0,
        step_px=1.0,
        query_source="real",
    )

    cpu_query, cpu_render, cpu_target, cpu_baseline, _ = rgb_patch_training._stack_patch_batch(
        **kwargs,
        query_cache={},
        render_cache={},
        image_cache_device=None,
    )
    cached_query, cached_render, cached_target, cached_baseline, _ = rgb_patch_training._stack_patch_batch(
        **kwargs,
        query_cache={},
        render_cache={},
        image_cache_device=torch.device("cpu"),
    )

    assert torch.allclose(cached_query.cpu(), cpu_query)
    assert torch.allclose(cached_render.cpu(), cpu_render)
    assert torch.allclose(cached_target.cpu(), cpu_target)
    assert torch.allclose(cached_baseline.cpu(), cpu_baseline)

    cpu_cache = rgb_patch_training.TensorImageLRUCache(
        max_bytes=1024 * 1024,
        storage_dtype=torch.float16,
    )
    streamed_query, streamed_render, streamed_target, streamed_baseline, _ = (
        rgb_patch_training._stack_patch_batch(
            **kwargs,
            query_cache=cpu_cache,
            render_cache=cpu_cache,
            image_cache_device=None,
            crop_device=torch.device("cpu"),
        )
    )
    assert all(cpu_cache[key].device.type == "cpu" for key in list(cpu_cache))
    assert torch.allclose(streamed_query, cpu_query, atol=1e-3)
    assert torch.allclose(streamed_render, cpu_render, atol=1e-3)
    assert torch.allclose(streamed_target, cpu_target)
    assert torch.allclose(streamed_baseline, cpu_baseline)


def test_continuous_offset_nll_can_upweight_dustbin_positive_targets() -> None:
    offsets = local_offset_grid(search_radius_px=1.0, step_px=1.0)
    logits = torch.zeros((2, int(offsets.shape[0])), dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float32)
    dustbin_logit = torch.zeros((2,), dtype=torch.float32)
    target_is_dustbin = torch.tensor([False, True])

    base_loss, _base_pred = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=dustbin_logit,
        search_radius_px=1.0,
        target_is_dustbin=target_is_dustbin,
        dustbin_bce_weight=1.0,
        dustbin_positive_weight=1.0,
    )
    weighted_loss, _weighted_pred = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=dustbin_logit,
        search_radius_px=1.0,
        target_is_dustbin=target_is_dustbin,
        dustbin_bce_weight=1.0,
        dustbin_positive_weight=5.0,
    )

    assert weighted_loss > base_loss


def test_continuous_offset_nll_dustbin_bce_weight_zero_disables_validity_term() -> None:
    offsets = local_offset_grid(search_radius_px=1.0, step_px=1.0)
    logits = torch.zeros((2, int(offsets.shape[0])), dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float32)
    target_is_dustbin = torch.tensor([False, True])

    low_loss, _low_pred = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=torch.full((2,), -10.0),
        search_radius_px=1.0,
        target_is_dustbin=target_is_dustbin,
        dustbin_bce_weight=0.0,
        dustbin_positive_weight=10.0,
    )
    high_loss, _high_pred = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=torch.full((2,), 10.0),
        search_radius_px=1.0,
        target_is_dustbin=target_is_dustbin,
        dustbin_bce_weight=0.0,
        dustbin_positive_weight=10.0,
    )

    assert torch.allclose(high_loss, low_loss)


def test_crop_rgb_window_with_affine_samples_source_from_output_offsets() -> None:
    image = torch.zeros((1, 3, 9, 9), dtype=torch.float32)
    image[:, 0] = torch.arange(9, dtype=torch.float32).reshape(1, 1, 9)
    source_from_output = torch.tensor([[[0.5, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    patch, offsets = crop_rgb_window_with_source_from_output_affine(
        image,
        torch.tensor([[4.0, 4.0]], dtype=torch.float32),
        source_from_output,
        radius_px=2.0,
        step_px=1.0,
        image_width=9,
        image_height=9,
    )

    assert offsets.shape == (25, 2)
    assert torch.allclose(patch[0, 0, 2, 2], torch.tensor(4.0), atol=1e-5)
    assert torch.allclose(patch[0, 0, 2, 4], torch.tensor(5.0), atol=1e-5)


def test_crop_rgb_window_with_homography_samples_absolute_output_to_source_pixels() -> None:
    image = torch.zeros((1, 3, 9, 9), dtype=torch.float32)
    image[:, 0] = torch.arange(9, dtype=torch.float32).reshape(1, 1, 9)
    source_from_output = torch.tensor([[[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)

    patch, offsets = crop_rgb_window_with_source_from_output_homography(
        image,
        torch.tensor([[8.0, 4.0]], dtype=torch.float32),
        source_from_output,
        radius_px=2.0,
        step_px=1.0,
        image_width=9,
        image_height=9,
    )

    assert offsets.shape == (25, 2)
    assert torch.allclose(patch[0, 0, 2, 2], torch.tensor(4.0), atol=1e-5)
    assert torch.allclose(patch[0, 0, 2, 4], torch.tensor(5.0), atol=1e-5)


def test_continuous_offset_nll_with_dustbin_handles_in_window_and_out_of_window_targets() -> None:
    offsets = local_offset_grid(search_radius_px=1.0, step_px=0.5)
    target = torch.tensor([[0.5, -0.5], [2.0, 0.0]], dtype=torch.float32)
    logits = -torch.sum((offsets[None] - target[:1, None]) ** 2, dim=2).repeat(2, 1) / 0.02
    dustbin_logit = torch.tensor([-10.0, 10.0], dtype=torch.float32)

    loss, pred = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=dustbin_logit,
        search_radius_px=1.0,
    )

    assert torch.isfinite(loss)
    assert pred.target_is_dustbin.tolist() == [False, True]
    assert pred.epe_px[0].item() < 0.1
    assert pred.dustbin_probability[1].item() > 0.99


def test_continuous_offset_nll_with_dustbin_supports_explicit_dustbin_bce_weight() -> None:
    offsets = local_offset_grid(search_radius_px=1.0, step_px=1.0)
    target = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    logits = torch.zeros((1, offsets.shape[0]), dtype=torch.float32)
    dustbin_logit = torch.tensor([-4.0], dtype=torch.float32)

    base_loss, _ = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=dustbin_logit,
        search_radius_px=1.0,
        dustbin_bce_weight=0.0,
    )
    weighted_loss, _ = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=dustbin_logit,
        search_radius_px=1.0,
        dustbin_bce_weight=2.0,
    )

    assert weighted_loss.item() > base_loss.item()


def test_continuous_offset_nll_with_dustbin_can_use_explicit_dustbin_targets_inside_window() -> None:
    offsets = local_offset_grid(search_radius_px=1.0, step_px=0.5)
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    logits = torch.zeros((2, offsets.shape[0]), dtype=torch.float32)
    dustbin_logit = torch.tensor([-10.0, 10.0], dtype=torch.float32)

    loss, pred = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=dustbin_logit,
        search_radius_px=1.0,
        target_is_dustbin=torch.tensor([False, True]),
    )

    assert torch.isfinite(loss)
    assert pred.target_is_dustbin.tolist() == [False, True]
    assert pred.dustbin_probability[1].item() > pred.dustbin_probability[0].item()


def test_continuous_offset_nll_can_use_gaussian_target_heatmap_for_continuous_supervision() -> None:
    offsets = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    good_logits = torch.tensor([[0.0, 4.0, 0.0]], dtype=torch.float32)
    bad_logits = torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float32)

    good_loss, _good_pred = continuous_offset_nll_with_dustbin(
        good_logits,
        offsets,
        target,
        dustbin_logit=torch.tensor([-10.0], dtype=torch.float32),
        search_radius_px=1.0,
        target_heatmap_sigma_px=0.5,
    )
    bad_loss, _bad_pred = continuous_offset_nll_with_dustbin(
        bad_logits,
        offsets,
        target,
        dustbin_logit=torch.tensor([-10.0], dtype=torch.float32),
        search_radius_px=1.0,
        target_heatmap_sigma_px=0.5,
    )

    assert good_loss.item() < bad_loss.item()


def test_continuous_offset_nll_honors_per_row_sample_weight() -> None:
    offsets = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    logits = torch.tensor([[0.0, 6.0, 0.0], [6.0, 0.0, 0.0]], dtype=torch.float32)
    dustbin = torch.tensor([-10.0, -10.0], dtype=torch.float32)

    unweighted_loss, _ = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=dustbin,
        search_radius_px=1.0,
    )
    weighted_loss, _ = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=dustbin,
        search_radius_px=1.0,
        sample_weight=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )

    assert weighted_loss.item() < unweighted_loss.item() * 0.1


def test_coarse_stage_likelihood_loss_supervises_two_stage_coarse_logits() -> None:
    coarse_offsets = local_offset_grid(search_radius_px=1.0, step_px=1.0)
    target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    dustbin = torch.tensor([-10.0], dtype=torch.float32)
    correct = RGBPatchMeasurementPrediction(
        logits=torch.empty((1, 0), dtype=torch.float32),
        offsets_xy=torch.empty((0, 2), dtype=torch.float32),
        dustbin_logit=dustbin,
        coarse_logits=-torch.sum((coarse_offsets[None] - target[:, None]) ** 2, dim=2) * 10.0,
        coarse_offsets_xy=coarse_offsets,
    )
    wrong = RGBPatchMeasurementPrediction(
        logits=torch.empty((1, 0), dtype=torch.float32),
        offsets_xy=torch.empty((0, 2), dtype=torch.float32),
        dustbin_logit=dustbin,
        coarse_logits=-torch.sum((coarse_offsets[None] - torch.tensor([[[-1.0, 0.0]]])) ** 2, dim=2) * 10.0,
        coarse_offsets_xy=coarse_offsets,
    )

    correct_loss, _ = rgb_patch_training._coarse_stage_likelihood_loss(
        correct,
        target,
        search_radius_px=1.0,
        target_is_dustbin=None,
        sample_weight=None,
        dustbin_positive_weight=1.0,
        target_heatmap_sigma_px=0.0,
    )
    wrong_loss, _ = rgb_patch_training._coarse_stage_likelihood_loss(
        wrong,
        target,
        search_radius_px=1.0,
        target_is_dustbin=None,
        sample_weight=None,
        dustbin_positive_weight=1.0,
        target_heatmap_sigma_px=0.0,
    )

    assert correct_loss.item() < wrong_loss.item()


def test_gate_supervision_loss_teaches_center_preserve_and_large_residual_correction() -> None:
    pred = RGBPatchMeasurementPrediction(
        logits=torch.empty((2, 0), dtype=torch.float32),
        offsets_xy=torch.empty((0, 2), dtype=torch.float32),
        dustbin_logit=torch.zeros((2,), dtype=torch.float32),
        gate_logit=torch.tensor([-4.0, 4.0], dtype=torch.float32),
    )
    target = torch.tensor([[0.25, 0.0], [2.0, 0.0]], dtype=torch.float32)

    good = rgb_patch_training._gate_supervision_loss(
        pred,
        target,
        target_is_dustbin=None,
        sample_weight=None,
        center_radius_px=0.5,
        full_radius_px=2.0,
    )
    bad = rgb_patch_training._gate_supervision_loss(
        RGBPatchMeasurementPrediction(
            logits=torch.empty((2, 0), dtype=torch.float32),
            offsets_xy=torch.empty((0, 2), dtype=torch.float32),
            dustbin_logit=torch.zeros((2,), dtype=torch.float32),
            gate_logit=torch.tensor([4.0, -4.0], dtype=torch.float32),
        ),
        target,
        target_is_dustbin=None,
        sample_weight=None,
        center_radius_px=0.5,
        full_radius_px=2.0,
    )

    assert good is not None
    assert bad is not None
    assert good.item() < bad.item()


def test_gate_supervision_loss_can_target_likelihood_utility() -> None:
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    good = rgb_patch_training._gate_supervision_loss(
        RGBPatchMeasurementPrediction(
            logits=torch.empty((2, 0), dtype=torch.float32),
            offsets_xy=torch.empty((0, 2), dtype=torch.float32),
            dustbin_logit=torch.zeros((2,), dtype=torch.float32),
            mean_offset_xy=torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32),
            gate_logit=torch.tensor([4.0, -4.0], dtype=torch.float32),
        ),
        target,
        target_is_dustbin=None,
        sample_weight=None,
        center_radius_px=0.5,
        full_radius_px=2.0,
        target_mode="utility",
        utility_temperature_px=0.1,
    )
    bad = rgb_patch_training._gate_supervision_loss(
        RGBPatchMeasurementPrediction(
            logits=torch.empty((2, 0), dtype=torch.float32),
            offsets_xy=torch.empty((0, 2), dtype=torch.float32),
            dustbin_logit=torch.zeros((2,), dtype=torch.float32),
            mean_offset_xy=torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32),
            gate_logit=torch.tensor([-4.0, 4.0], dtype=torch.float32),
        ),
        target,
        target_is_dustbin=None,
        sample_weight=None,
        center_radius_px=0.5,
        full_radius_px=2.0,
        target_mode="utility",
        utility_temperature_px=0.1,
    )

    assert good is not None
    assert bad is not None
    assert good.item() < bad.item()


def test_dustbin_probability_is_independent_of_spatial_softmax_scale() -> None:
    offsets = local_offset_grid(search_radius_px=1.0, step_px=0.5)
    target = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    logits = torch.full((1, offsets.shape[0]), 100.0, dtype=torch.float32)

    _loss, pred = continuous_offset_nll_with_dustbin(
        logits,
        offsets,
        target,
        dustbin_logit=torch.tensor([10.0], dtype=torch.float32),
        search_radius_px=1.0,
        target_is_dustbin=torch.tensor([True]),
    )

    assert pred.dustbin_probability.item() > 0.99


def test_rgb_patch_measurement_branch_outputs_dense_offset_likelihood() -> None:
    branch = RGBPatchMeasurementBranch(search_radius_px=1.0, context_radius_px=1.0, step_px=0.5, feature_dim=8)
    query_patch = torch.rand((2, 3, 9, 9), dtype=torch.float32)
    render_patch = torch.rand((2, 3, 9, 9), dtype=torch.float32)

    pred = branch.forward_from_patches(query_patch, render_patch)

    assert pred.logits.shape == (2, 25)
    assert pred.offsets_xy.shape == (25, 2)
    assert pred.dustbin_logit.shape == (2,)
    assert pred.direct_mean_offset_xy is not None
    assert pred.direct_log_sigma_xy is not None
    assert pred.direct_mean_offset_xy.shape == (2, 2)
    assert pred.gated_mean_offset_xy is not None
    assert pred.gate_probability is not None
    assert pred.gated_mean_offset_xy.shape == (2, 2)
    assert pred.gate_probability.shape == (2,)
    assert torch.all(pred.gate_probability >= 0.0)
    assert torch.all(pred.gate_probability <= 1.0)


def test_texture_patch_encoder_fpn_arch_outputs_dense_normalized_features() -> None:
    encoder = TexturePatchEncoder(feature_dim=12, hidden_dim=16, input_mode="norm_graygrad", encoder_arch="fpn")
    patch = torch.rand((2, 3, 17, 17), dtype=torch.float32)

    feat = encoder(patch)

    assert feat.shape == (2, 12, 17, 17)
    assert torch.all(torch.isfinite(feat))
    norms = torch.linalg.norm(feat, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_rgb_patch_measurement_branch_accepts_fpn_encoder_arch() -> None:
    branch = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=0.5,
        feature_dim=12,
        hidden_dim=16,
        input_mode="norm_graygrad",
        encoder_arch="fpn",
    )
    query_patch = torch.rand((2, 3, 9, 9), dtype=torch.float32)
    render_patch = torch.rand((2, 3, 9, 9), dtype=torch.float32)

    pred = branch.forward_from_patches(query_patch, render_patch)

    assert pred.logits.shape == (2, 25)
    assert pred.mean_offset_xy is not None
    assert pred.gated_mean_offset_xy is not None


def test_rgb_patch_measurement_branch_accepts_template_scale_factors() -> None:
    branch = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=2.0,
        step_px=1.0,
        feature_dim=8,
        template_scale_factors=(0.75, 1.0, 1.25),
    )
    query_patch = torch.rand((2, 3, 7, 7), dtype=torch.float32)
    render_patch = torch.rand((2, 3, 7, 7), dtype=torch.float32)

    pred = branch.forward_from_patches(query_patch, render_patch)

    assert pred.logits.shape == (2, 9)
    assert pred.offsets_xy.shape == (9, 2)
    assert branch.template_scale_factors == (0.75, 1.0, 1.25)


def test_rgb_patch_measurement_branch_can_use_two_stage_coarse_to_fine_search() -> None:
    branch = RGBPatchMeasurementBranch(
        coarse_search_radius_px=2.0,
        coarse_step_px=1.0,
        search_radius_px=0.5,
        context_radius_px=1.0,
        step_px=0.5,
        feature_dim=8,
    )
    query_patch = torch.rand((2, 3, 15, 15), dtype=torch.float32)
    render_patch = torch.rand((2, 3, 15, 15), dtype=torch.float32)

    pred = branch.forward_from_patches(query_patch, render_patch)

    assert branch.crop_radius_px == 3.5
    assert branch.measurement_search_radius_px == 2.5
    assert pred.logits.shape == (2, 9)
    assert pred.offsets_xy.shape == (2, 9, 2)
    assert pred.coarse_logits is not None
    assert pred.coarse_offsets_xy is not None
    assert pred.coarse_mode_offset_xy is not None
    assert pred.coarse_logits.shape == (2, 25)
    assert pred.coarse_offsets_xy.shape == (25, 2)
    assert pred.coarse_mode_offset_xy.shape == (2, 2)


def test_patch_forward_only_preserves_two_stage_coarse_logits_for_training() -> None:
    branch = RGBPatchMeasurementBranch(
        coarse_search_radius_px=2.0,
        coarse_step_px=1.0,
        search_radius_px=0.5,
        context_radius_px=1.0,
        step_px=0.5,
        feature_dim=8,
    )
    wrapper = rgb_patch_training._PatchForwardOnly(branch)
    query_patch = torch.rand((2, 3, 15, 15), dtype=torch.float32)
    render_patch = torch.rand((2, 3, 15, 15), dtype=torch.float32)

    (
        logits,
        offsets,
        dustbin,
        likelihood_mean,
        direct_mean,
        direct_log_sigma,
        gated_mean,
        gate_logit,
        coarse_logits,
        coarse_offsets,
    ) = wrapper(
        query_patch,
        render_patch,
        torch.empty((0,), dtype=torch.float32),
    )

    assert logits.shape == (2, 9)
    assert offsets.shape == (2, 9, 2)
    assert dustbin.shape == (2,)
    assert likelihood_mean.shape == (2, 2)
    assert direct_mean.shape == (2, 2)
    assert direct_log_sigma.shape == (2, 2)
    assert gated_mean.shape == (2, 2)
    assert gate_logit.shape == (2,)
    assert coarse_logits is not None
    assert coarse_offsets is not None
    assert coarse_logits.shape == (2, 25)
    assert coarse_offsets.shape == (2, 25, 2)


def test_patch_forward_only_expands_single_stage_offsets_for_data_parallel_gather() -> None:
    branch = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=0.5,
        feature_dim=8,
    )
    wrapper = rgb_patch_training._PatchForwardOnly(branch)
    query_patch = torch.rand((2, 3, 9, 9), dtype=torch.float32)
    render_patch = torch.rand((2, 3, 9, 9), dtype=torch.float32)

    (
        logits,
        offsets,
        _dustbin,
        likelihood_mean,
        _direct_mean,
        _direct_log_sigma,
        _gated_mean,
        _gate_logit,
        _coarse_logits,
        _coarse_offsets,
    ) = wrapper(
        query_patch,
        render_patch,
        torch.empty((0,), dtype=torch.float32),
    )

    assert logits.shape == (2, 25)
    assert offsets.shape == (2, 25, 2)
    assert likelihood_mean.shape == (2, 2)


def test_rgb_patch_measurement_branch_can_condition_logits_on_prior_scale() -> None:
    branch = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=2.0,
        step_px=1.0,
        feature_dim=8,
        condition_on_prior_scale=True,
    )
    query_patch = torch.rand((2, 3, 7, 7), dtype=torch.float32)
    render_patch = torch.rand((2, 3, 7, 7), dtype=torch.float32)

    pred_small = branch.forward_from_patches(query_patch, render_patch, prior_scale_px=torch.tensor([0.5, 0.5]))
    pred_large = branch.forward_from_patches(query_patch, render_patch, prior_scale_px=torch.tensor([1.0, 1.0]))

    assert pred_small.logits.shape == pred_large.logits.shape == (2, 9)
    assert not torch.allclose(pred_small.logits, pred_large.logits)


def test_rgb_patch_measurement_branch_prior_scale_experts_select_different_logit_biases() -> None:
    branch = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        condition_on_prior_scale=True,
        prior_scale_expert_centers_px=(0.5, 2.0),
    )
    query_patch = torch.rand((1, 3, 5, 5), dtype=torch.float32)
    render_patch = torch.rand((1, 3, 5, 5), dtype=torch.float32)
    with torch.no_grad():
        branch.prior_scale_expert_logit_bias.zero_()
        branch.prior_scale_expert_logit_bias[0, 0] = 100.0
        branch.prior_scale_expert_logit_bias[1, -1] = 100.0

    pred_small = branch.forward_from_patches(query_patch, render_patch, prior_scale_px=torch.tensor([0.5]))
    pred_large = branch.forward_from_patches(query_patch, render_patch, prior_scale_px=torch.tensor([2.0]))

    assert int(torch.argmax(pred_small.logits[0]).item()) == 0
    assert int(torch.argmax(pred_large.logits[0]).item()) == int(pred_large.logits.shape[1] - 1)


def test_rgb_patch_measurement_branch_prior_scale_experts_use_independent_feature_projections() -> None:
    branch = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        condition_on_prior_scale=True,
        prior_scale_expert_centers_px=(0.5, 2.0),
        prior_scale_expert_projection=True,
    )
    query_patch = torch.rand((1, 3, 5, 5), dtype=torch.float32)
    render_patch = torch.rand((1, 3, 5, 5), dtype=torch.float32)
    with torch.no_grad():
        branch.prior_scale_expert_logit_bias.zero_()
        branch.prior_scale_expert_query_projections[0].weight.zero_()
        branch.prior_scale_expert_render_projections[0].weight.zero_()
        branch.prior_scale_expert_query_projections[1].weight.zero_()
        branch.prior_scale_expert_render_projections[1].weight.zero_()
        for channel in range(8):
            branch.prior_scale_expert_query_projections[0].weight[channel, channel, 0, 0] = 1.0
            branch.prior_scale_expert_render_projections[0].weight[channel, channel, 0, 0] = 1.0

    pred_identity = branch.forward_from_patches(query_patch, render_patch, prior_scale_px=torch.tensor([0.5]))
    pred_zero = branch.forward_from_patches(query_patch, render_patch, prior_scale_px=torch.tensor([2.0]))

    assert pred_identity.logits.shape == pred_zero.logits.shape == (1, 9)
    assert torch.max(torch.abs(pred_zero.logits)).item() < torch.max(torch.abs(pred_identity.logits)).item()


def test_rgb_patch_measurement_branch_prior_scale_experts_can_use_hard_nearest_gate() -> None:
    branch = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        condition_on_prior_scale=True,
        prior_scale_expert_centers_px=(0.5, 2.0),
        prior_scale_expert_projection=True,
        prior_scale_expert_gate="hard",
    )
    query_patch = torch.rand((1, 3, 5, 5), dtype=torch.float32)
    render_patch = torch.rand((1, 3, 5, 5), dtype=torch.float32)
    with torch.no_grad():
        for parameter in branch.prior_scale_logit_bias.parameters():
            parameter.zero_()
        branch.prior_scale_expert_logit_bias.zero_()
        for projection in list(branch.prior_scale_expert_query_projections) + list(branch.prior_scale_expert_render_projections):
            projection.weight.zero_()
        for channel in range(8):
            branch.prior_scale_expert_query_projections[0].weight[channel, channel, 0, 0] = 1.0
            branch.prior_scale_expert_render_projections[0].weight[channel, channel, 0, 0] = 1.0

    pred_zero = branch.forward_from_patches(query_patch, render_patch, prior_scale_px=torch.tensor([2.0]))

    assert torch.max(torch.abs(pred_zero.logits)).item() < 1e-6


def test_rgb_patch_measurement_branch_supports_gray_gradient_input_mode() -> None:
    branch = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=0.5,
        feature_dim=8,
        input_mode="rgb_graygrad",
    )
    query_patch = torch.rand((2, 3, 9, 9), dtype=torch.float32)
    render_patch = torch.rand((2, 3, 9, 9), dtype=torch.float32)

    pred = branch.forward_from_patches(query_patch, render_patch)

    assert pred.logits.shape == (2, 25)
    assert pred.offsets_xy.shape == (25, 2)
    assert branch.encoder.input_channels == 6


def test_texture_encoder_norm_graygrad_input_is_affine_brightness_invariant() -> None:
    branch = RGBPatchMeasurementBranch(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=0.5,
        feature_dim=8,
        input_mode="norm_graygrad",
    )
    patch = torch.linspace(0.1, 0.8, 9 * 9, dtype=torch.float32).reshape(1, 1, 9, 9).repeat(1, 3, 1, 1)
    shifted = patch * 0.5 + 0.2

    prepared = branch.encoder._prepare_input(patch)
    prepared_shifted = branch.encoder._prepare_input(shifted)

    assert branch.encoder.input_channels == 3
    assert prepared.shape == (1, 3, 9, 9)
    assert torch.allclose(prepared, prepared_shifted, atol=1e-5)


def test_rgb_patch_measurement_branch_dustbin_logit_depends_on_patch_pair() -> None:
    torch.manual_seed(5)
    branch = RGBPatchMeasurementBranch(search_radius_px=1.0, context_radius_px=1.0, step_px=0.5, feature_dim=8)
    render_patch = torch.rand((2, 3, 9, 9), dtype=torch.float32)
    query_patch = render_patch.clone()
    query_patch[1].zero_()

    pred = branch.forward_from_patches(query_patch, render_patch)

    assert pred.dustbin_logit.shape == (2,)
    assert not torch.allclose(pred.dustbin_logit[0], pred.dustbin_logit[1])


def test_stack_patch_batch_can_train_against_policy_target_columns(tmp_path: Path) -> None:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[8, 8] = [255, 255, 255]
    render_cache = tmp_path / "render_rgb_depth" / "r0.npz"
    render_cache.parent.mkdir(parents=True)
    np.savez_compressed(render_cache, rgb=rgb, depth=np.ones((16, 16), dtype=np.float32), alpha=np.ones((16, 16), dtype=np.float32))
    row = {
        "query_id": "seq0/frame00000.png",
        "render_x": 8.0,
        "render_y": 8.0,
        "center_x": 7.5,
        "center_y": 8.0,
        "query_gt_x": 9.0,
        "query_gt_y": 8.0,
        "policy_target_x": 7.25,
        "policy_target_y": 8.75,
    }

    _query_patch, _render_patch, default_target, default_baseline, _default_dustbin = rgb_patch_training._stack_patch_batch(
        [row],
        image_root=tmp_path,
        render_cache_by_query={"seq0/frame00000.png": render_cache},
        image_width=16,
        image_height=16,
        crop_radius_px=2.0,
        step_px=1.0,
        query_cache={},
        render_cache={},
        query_source="render",
    )
    _query_patch, _render_patch, policy_target, policy_baseline, _policy_dustbin = rgb_patch_training._stack_patch_batch(
        [row],
        image_root=tmp_path,
        render_cache_by_query={"seq0/frame00000.png": render_cache},
        image_width=16,
        image_height=16,
        crop_radius_px=2.0,
        step_px=1.0,
        query_cache={},
        render_cache={},
        query_source="render",
        target_x_key="policy_target_x",
        target_y_key="policy_target_y",
    )

    assert torch.allclose(default_target, torch.tensor([[1.5, 0.0]], dtype=torch.float32))
    assert torch.allclose(default_baseline, torch.tensor([1.5], dtype=torch.float32))
    assert torch.allclose(policy_target, torch.tensor([[-0.25, 0.75]], dtype=torch.float32))
    assert torch.allclose(policy_baseline, torch.tensor([float(np.hypot(0.25, 0.75))], dtype=torch.float32))


def test_sample_weight_batch_reads_opt_in_loss_weight_column() -> None:
    rows = [{"policy_loss_weight": "1.0"}, {"policy_loss_weight": "0.25"}]

    assert rgb_patch_training._sample_weight_batch(rows, loss_weight_key="") is None
    assert torch.allclose(
        rgb_patch_training._sample_weight_batch(rows, loss_weight_key="policy_loss_weight"),
        torch.tensor([1.0, 0.25], dtype=torch.float32),
    )

    try:
        rgb_patch_training._sample_weight_batch([{}], loss_weight_key="policy_loss_weight")
    except ValueError as exc:
        assert "missing 'policy_loss_weight'" in str(exc)
    else:
        raise AssertionError("expected missing loss weight column to fail")


def test_template_search_cost_volume_peaks_at_shifted_template_offset() -> None:
    query_features = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    render_features = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    template = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    render_features[0, 0, 1:4, 1:4] = template
    query_features[0, 0, 1:4, 2:5] = template

    logits, offsets = template_search_cost_volume_logits(
        query_features,
        render_features,
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        temperature=10.0,
    )

    assert logits.shape == (1, 9)
    assert torch.allclose(offsets[torch.argmax(logits, dim=1).item()], torch.tensor([1.0, 0.0]))


def test_template_search_cost_volume_supports_multiple_template_scales() -> None:
    query_features = torch.rand((2, 3, 7, 7), dtype=torch.float32, requires_grad=True)
    render_features = torch.rand((2, 3, 7, 7), dtype=torch.float32, requires_grad=True)

    logits, offsets = template_search_cost_volume_logits(
        query_features,
        render_features,
        search_radius_px=1.0,
        context_radius_px=2.0,
        step_px=1.0,
        temperature=5.0,
        template_scale_factors=(0.75, 1.0, 1.25),
    )
    loss = logits.mean()
    loss.backward()

    assert logits.shape == (2, 9)
    assert offsets.shape == (9, 2)
    assert query_features.grad is not None
    assert render_features.grad is not None


def test_template_search_cost_volume_can_decouple_feature_and_output_steps() -> None:
    query_features = torch.zeros((1, 1, 7, 7), dtype=torch.float32)
    render_features = torch.zeros((1, 1, 7, 7), dtype=torch.float32)
    template = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    render_features[0, 0, 2:5, 2:5] = template
    query_features[0, 0, 2:5, 4:7] = template

    logits, offsets = template_search_cost_volume_logits(
        query_features,
        render_features,
        search_radius_px=2.0,
        context_radius_px=1.0,
        step_px=1.0,
        output_step_px=2.0,
        temperature=10.0,
    )

    assert logits.shape == (1, 9)
    assert offsets.shape == (9, 2)
    assert torch.allclose(offsets[torch.argmax(logits, dim=1).item()], torch.tensor([2.0, 0.0]))


def test_coarse_to_fine_template_search_volume_centers_fine_grid_on_coarse_mode() -> None:
    query_features = torch.zeros((1, 1, 11, 11), dtype=torch.float32)
    render_features = torch.zeros((1, 1, 11, 11), dtype=torch.float32)
    template = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    render_features[0, 0, 4:7, 4:7] = template
    query_features[0, 0, 5:8, 6:9] = template

    fine_logits, fine_offsets, coarse_logits, coarse_offsets = coarse_to_fine_template_search_cost_volume_logits(
        query_features,
        render_features,
        coarse_search_radius_px=3.0,
        coarse_step_px=1.0,
        fine_search_radius_px=1.0,
        fine_step_px=1.0,
        context_radius_px=1.0,
        feature_step_px=1.0,
        temperature=10.0,
    )

    assert coarse_logits.shape == (1, 49)
    assert coarse_offsets.shape == (49, 2)
    assert torch.allclose(coarse_offsets[torch.argmax(coarse_logits, dim=1).item()], torch.tensor([2.0, 1.0]))
    assert fine_logits.shape == (1, 9)
    assert fine_offsets.shape == (1, 9, 2)
    assert torch.allclose(fine_offsets[0, torch.argmax(fine_logits, dim=1).item()], torch.tensor([2.0, 1.0]))


def test_cost_volume_quality_features_capture_peakiness_without_absolute_logit_scale() -> None:
    flat = torch.full((1, 9), 10.0, dtype=torch.float32)
    peaked = flat.clone()
    peaked[0, 4] = 12.0
    shifted = peaked + 100.0

    flat_quality = cost_volume_quality_features(flat)
    peaked_quality = cost_volume_quality_features(peaked)
    shifted_quality = cost_volume_quality_features(shifted)

    assert flat_quality.shape == (1, 6)
    assert peaked_quality[0, 0] > flat_quality[0, 0]
    assert peaked_quality[0, 1] > flat_quality[0, 1]
    assert peaked_quality[0, 3] < flat_quality[0, 3]
    assert torch.allclose(peaked_quality[:, :5], shifted_quality[:, :5], atol=1e-5)


def test_residual_delta_gaussian_nll_prefers_correct_delta_and_reports_epe() -> None:
    target = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    correct = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    wrong = torch.tensor([[-0.5, 0.25]], dtype=torch.float32)
    log_sigma = torch.zeros((1, 2), dtype=torch.float32)

    correct_loss, correct_pred = residual_delta_gaussian_nll(correct, log_sigma, target, search_radius_px=1.0)
    wrong_loss, wrong_pred = residual_delta_gaussian_nll(wrong, log_sigma, target, search_radius_px=1.0)

    assert correct_loss.item() < wrong_loss.item()
    assert correct_pred.epe_px is not None
    assert correct_pred.epe_px.item() < 1e-6


def test_residual_delta_gaussian_nll_penalizes_high_dustbin_on_valid_targets() -> None:
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    mean = target.clone()
    log_sigma = torch.zeros((2, 2), dtype=torch.float32)

    low_dustbin_loss, low_pred = residual_delta_gaussian_nll(
        mean,
        log_sigma,
        target,
        dustbin_logit=torch.tensor([-10.0, -10.0], dtype=torch.float32),
        search_radius_px=1.0,
        target_is_dustbin=torch.tensor([False, False]),
    )
    high_dustbin_loss, high_pred = residual_delta_gaussian_nll(
        mean,
        log_sigma,
        target,
        dustbin_logit=torch.tensor([10.0, 10.0], dtype=torch.float32),
        search_radius_px=1.0,
        target_is_dustbin=torch.tensor([False, False]),
    )

    assert low_pred.target_is_dustbin.tolist() == [False, False]
    assert high_pred.target_is_dustbin.tolist() == [False, False]
    assert high_dustbin_loss.item() > low_dustbin_loss.item() + 5.0


def test_residual_delta_gaussian_nll_honors_per_row_sample_weight() -> None:
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    mean = torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    log_sigma = torch.zeros((2, 2), dtype=torch.float32)

    unweighted_loss, _ = residual_delta_gaussian_nll(mean, log_sigma, target, search_radius_px=4.0)
    weighted_loss, _ = residual_delta_gaussian_nll(
        mean,
        log_sigma,
        target,
        search_radius_px=4.0,
        sample_weight=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )

    assert weighted_loss.item() < unweighted_loss.item() * 0.1


def test_augment_render_template_patch_keeps_shape_and_changes_realistic_mode() -> None:
    patch = torch.full((2, 3, 5, 5), 0.5, dtype=torch.float32)
    torch.manual_seed(7)

    augmented = augment_render_template_patch(patch, mode="realistic")

    assert augmented.shape == patch.shape
    assert torch.all(augmented >= 0.0)
    assert torch.all(augmented <= 1.0)
    assert not torch.allclose(augmented, patch)
    assert torch.allclose(augment_render_template_patch(patch, mode="none"), patch)


def test_binary_auroc_reports_separable_and_tied_scores() -> None:
    assert rgb_patch_training._binary_auroc(
        torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=torch.float32),
        torch.tensor([False, False, True, True]),
    ) == 1.0
    assert rgb_patch_training._binary_auroc(
        torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32),
        torch.tensor([False, False, True, True]),
    ) == 0.5
    assert rgb_patch_training._binary_auroc(torch.tensor([0.1, 0.2]), torch.tensor([False, False])) is None


def test_train_rgb_patch_measurement_branch_reports_center_baseline_and_improve_ratio(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[8, 8] = [255, 255, 255]
    Image.fromarray(rgb).save(image_root / "seq0" / "frame00000.png")
    render_cache = tmp_path / "render_rgb_depth" / "r0.npz"
    render_cache.parent.mkdir(parents=True)
    np.savez_compressed(render_cache, rgb=rgb, depth=np.ones((16, 16), dtype=np.float32), alpha=np.ones((16, 16), dtype=np.float32))
    manifest = tmp_path / "render_manifest.csv"
    _write_csv(manifest, [{"query_id": "seq0/frame00000.png", "rgb_depth_cache_path": str(render_cache)}])
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {
                "query_id": "seq0/frame00000.png",
                "render_x": 8.0,
                "render_y": 8.0,
                "center_x": 7.5,
                "center_y": 8.0,
                "query_gt_x": 8.0,
                "query_gt_y": 8.0,
            }
        ],
    )

    summary = train_rgb_patch_measurement_branch(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "train",
        image_width=16,
        image_height=16,
        search_radius_px=1.0,
        context_radius_px=2.0,
        step_px=0.5,
        steps=2,
        batch_size=1,
        feature_dim=4,
        device="cpu",
        query_source="render",
        template_scale_factors=(0.75, 1.0, 1.25),
    )

    assert summary["train_center_baseline"]["epe_px"] == 0.5
    assert summary["query_source"] == "render"
    assert summary["support_patch_warp"] == "none"
    assert summary["support_patch_source_audit"]["support_patch_source"] == "render_cache_by_query"
    assert summary["template_scale_factors"] == [0.75, 1.0, 1.25]
    assert summary["delta_loss_weight"] == 1.0
    assert "improve_ratio" in summary["val_metrics"]
    assert "valid_improve_ratio" in summary["val_metrics"]
    assert "likelihood_valid_improve_ratio" in summary["val_metrics"]
    assert "dustbin_count" in summary["val_metrics"]
    assert "valid_accept_recall_0p5" in summary["val_metrics"]
    assert "validity_accuracy_0p5" in summary["val_metrics"]
    assert "validity_brier" in summary["val_metrics"]
    assert "validity_auroc" in summary["val_metrics"]
    assert "dustbin_valid_probability_gap" in summary["val_metrics"]
    assert "accepted_count_0p5" in summary["val_metrics"]
    assert "accepted_valid_count_0p5" in summary["val_metrics"]
    assert "accepted_dustbin_count_0p5" in summary["val_metrics"]
    assert "accepted_valid_likelihood_epe_median_px_0p5" in summary["val_metrics"]
    assert "accepted_valid_likelihood_recall_0p5px_0p5" in summary["val_metrics"]
    assert "accepted_valid_likelihood_improve_ratio_0p5" in summary["val_metrics"]
    assert "dustbin_reject_recall_0p5" in summary["val_metrics"]
    assert summary["acceptance_gate"]["median_epe_threshold_px"] == 0.5
    assert summary["acceptance_gate"]["improve_ratio_threshold"] == 0.8
    assert "direct_head" in summary["acceptance_gate"]
    assert "likelihood_head" in summary["acceptance_gate"]
    assert "residual_bin_gates" in summary["acceptance_gate"]
    assert "likelihood_head" in summary["acceptance_gate"]["residual_bin_gates"]
    assert "passes" in summary["acceptance_gate"]["direct_head"]
    assert "passes" in summary["acceptance_gate"]["likelihood_head"]
    assert (tmp_path / "train" / "rgb_patch_measurement_branch.pt").exists()

    resumed = train_rgb_patch_measurement_branch(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "resume",
        image_width=16,
        image_height=16,
        search_radius_px=1.0,
        context_radius_px=2.0,
        step_px=0.5,
        steps=1,
        batch_size=1,
        feature_dim=4,
        device="cpu",
        query_source="render",
        init_checkpoint=tmp_path / "train" / "rgb_patch_measurement_branch.pt",
    )

    assert resumed["init_checkpoint"] == str(tmp_path / "train" / "rgb_patch_measurement_branch.pt")

    augmented = train_rgb_patch_measurement_branch(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "augmented",
        image_width=16,
        image_height=16,
        search_radius_px=1.0,
        context_radius_px=2.0,
        step_px=0.5,
        steps=1,
        batch_size=1,
        feature_dim=4,
        device="cpu",
        query_source="render_augmented",
        train_stage="render_augmented",
    )

    assert augmented["query_source"] == "render_augmented"
    assert augmented["train_stage"] == "render_augmented"

    dustbin_only = train_rgb_patch_measurement_branch(
        rows_csv=rows_csv,
        render_cache_manifest_csv=manifest,
        image_root=image_root,
        output_dir=tmp_path / "dustbin_only",
        image_width=16,
        image_height=16,
        search_radius_px=1.0,
        context_radius_px=2.0,
        step_px=0.5,
        steps=1,
        batch_size=1,
        feature_dim=4,
        device="cpu",
        query_source="render",
        train_dustbin_head_only=True,
    )

    assert dustbin_only["train_dustbin_head_only"] is True
    assert dustbin_only["trainable_parameter_count"] < dustbin_only["parameter_count"]


def test_stack_patch_batch_real_pair_uses_support_image_as_template(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    query_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    query_rgb[8, 9] = [255, 0, 0]
    support_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    support_rgb[8, 8] = [0, 255, 0]
    Image.fromarray(query_rgb).save(image_root / "seq0" / "query.png")
    Image.fromarray(support_rgb).save(image_root / "seq0" / "support.png")

    query_patch, support_patch, target, baseline, target_is_dustbin = rgb_patch_training._stack_patch_batch(
        [
            {
                "query_id": "seq0/query.png",
                "support_image_id": "seq0/support.png",
                "support_x": 8.0,
                "support_y": 8.0,
                "render_x": 8.0,
                "render_y": 8.0,
                "center_x": 8.0,
                "center_y": 8.0,
                "query_gt_x": 9.0,
                "query_gt_y": 8.0,
                "target_is_dustbin": "True",
            }
        ],
        image_root=image_root,
        render_cache_by_query={},
        image_width=16,
        image_height=16,
        crop_radius_px=1.0,
        step_px=1.0,
        query_cache={},
        render_cache={},
        query_source="real_pair",
    )

    assert torch.allclose(query_patch[0, :, 1, 1], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(query_patch[0, :, 1, 2], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(support_patch[0, :, 1, 1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(target, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(baseline, torch.tensor([1.0]))
    assert target_is_dustbin.tolist() == [True]


def test_stack_patch_batch_can_warp_support_patch_with_local_affine(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    query_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    support_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    for x in range(16):
        support_rgb[:, x, 0] = x * 10
    Image.fromarray(query_rgb).save(image_root / "seq0" / "query.png")
    Image.fromarray(support_rgb).save(image_root / "seq0" / "support.png")

    _query_patch, support_patch, _target, _baseline, _target_is_dustbin = rgb_patch_training._stack_patch_batch(
        [
            {
                "query_id": "seq0/query.png",
                "support_image_id": "seq0/support.png",
                "support_x": 8.0,
                "support_y": 8.0,
                "render_x": 8.0,
                "render_y": 8.0,
                "center_x": 8.0,
                "center_y": 8.0,
                "query_gt_x": 8.0,
                "query_gt_y": 8.0,
                "support_to_query_a00": 2.0,
                "support_to_query_a01": 0.0,
                "support_to_query_a10": 0.0,
                "support_to_query_a11": 1.0,
            }
        ],
        image_root=image_root,
        render_cache_by_query={},
        image_width=16,
        image_height=16,
        crop_radius_px=2.0,
        step_px=1.0,
        query_cache={},
        render_cache={},
        query_source="real_pair",
        support_patch_warp="local_affine",
    )

    assert torch.allclose(support_patch[0, 0, 2, 2], torch.tensor(80.0 / 255.0), atol=1e-5)
    assert torch.allclose(support_patch[0, 0, 2, 4], torch.tensor(90.0 / 255.0), atol=1e-5)


def test_stack_patch_batch_local_affine_falls_back_to_unwarped_patch_for_dustbin_without_affine(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    query_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    support_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    support_rgb[8, 8] = np.asarray([255, 0, 0], dtype=np.uint8)
    Image.fromarray(query_rgb).save(image_root / "seq0" / "query.png")
    Image.fromarray(support_rgb).save(image_root / "seq0" / "support.png")

    _query_patch, support_patch, _target, _baseline, target_is_dustbin = rgb_patch_training._stack_patch_batch(
        [
            {
                "query_id": "seq0/query.png",
                "support_image_id": "seq0/support.png",
                "support_x": 8.0,
                "support_y": 8.0,
                "render_x": 8.0,
                "render_y": 8.0,
                "center_x": 8.0,
                "center_y": 8.0,
                "query_gt_x": 9.0,
                "query_gt_y": 8.0,
                "target_is_dustbin": "True",
            }
        ],
        image_root=image_root,
        render_cache_by_query={},
        image_width=16,
        image_height=16,
        crop_radius_px=1.0,
        step_px=1.0,
        query_cache={},
        render_cache={},
        query_source="real_pair",
        support_patch_warp="local_affine",
    )

    assert target_is_dustbin.tolist() == [True]
    assert torch.allclose(support_patch[0, :, 1, 1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)


def test_stack_patch_batch_can_warp_support_patch_with_local_homography(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "seq0").mkdir(parents=True)
    query_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    support_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    for x in range(16):
        support_rgb[:, x, 0] = x * 10
    Image.fromarray(query_rgb).save(image_root / "seq0" / "query.png")
    Image.fromarray(support_rgb).save(image_root / "seq0" / "support.png")

    _query_patch, support_patch, _target, _baseline, _target_is_dustbin = rgb_patch_training._stack_patch_batch(
        [
            {
                "query_id": "seq0/query.png",
                "support_image_id": "seq0/support.png",
                "support_x": 4.0,
                "support_y": 4.0,
                "render_x": 4.0,
                "render_y": 4.0,
                "center_x": 8.0,
                "center_y": 4.0,
                "query_gt_x": 8.0,
                "query_gt_y": 4.0,
                "support_to_query_h00": 2.0,
                "support_to_query_h01": 0.0,
                "support_to_query_h02": 0.0,
                "support_to_query_h10": 0.0,
                "support_to_query_h11": 1.0,
                "support_to_query_h12": 0.0,
                "support_to_query_h20": 0.0,
                "support_to_query_h21": 0.0,
                "support_to_query_h22": 1.0,
            }
        ],
        image_root=image_root,
        render_cache_by_query={},
        image_width=16,
        image_height=16,
        crop_radius_px=2.0,
        step_px=1.0,
        query_cache={},
        render_cache={},
        query_source="real_pair",
        support_patch_warp="local_homography",
    )

    assert torch.allclose(support_patch[0, 0, 2, 2], torch.tensor(40.0 / 255.0), atol=1e-5)
    assert torch.allclose(support_patch[0, 0, 2, 4], torch.tensor(50.0 / 255.0), atol=1e-5)
