import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from feature_field.dcff.losses import DCFFLoss


def test_adaptive_compressor_pca_initialization_matches_centered_projection():
    from feature_field.dcff.radio_teacher import AdaptiveRadioCompressor

    components = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    mean = torch.tensor([0.25, -0.5, 1.0, 2.0])
    compressor = AdaptiveRadioCompressor(
        input_dim=4,
        output_dim=2,
        hidden_dim=8,
        pca_state={"components": components, "mean": mean},
        normalize_output=False,
    )

    raw = torch.randn(2, 4, 3, 5)
    projected = compressor(raw)
    expected = torch.einsum(
        "bchw,oc->bohw",
        raw - mean.view(1, 4, 1, 1),
        components[:2],
    )

    assert torch.allclose(projected, expected, atol=1e-5)


def test_adaptive_compressor_reconstruction_loss_is_differentiable():
    from feature_field.dcff.radio_teacher import AdaptiveRadioCompressor

    compressor = AdaptiveRadioCompressor(
        input_dim=8,
        output_dim=3,
        hidden_dim=16,
        sample_pixels=5,
    )
    raw = torch.randn(2, 8, 4, 4)
    z = compressor(raw)
    losses = compressor.reconstruction_loss(raw, z)

    assert {"cos", "l1", "std", "decorrelation", "total"} == set(losses)
    assert losses["total"].requires_grad
    losses["total"].backward()
    assert any(p.grad is not None for p in compressor.parameters())


def test_adaptive_compressor_reconstruction_loss_supports_chunking():
    from feature_field.dcff.radio_teacher import AdaptiveRadioCompressor

    compressor = AdaptiveRadioCompressor(
        input_dim=8,
        output_dim=3,
        hidden_dim=16,
        sample_pixels=13,
        recon_chunk_pixels=5,
    )
    raw = torch.randn(2, 8, 4, 4)
    z = compressor(raw)
    losses = compressor.reconstruction_loss(raw, z)

    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(p.grad is not None for p in compressor.decoder.parameters())


def test_adaptive_compressor_regularizes_low_variance_and_channel_correlation():
    from feature_field.dcff.radio_teacher import AdaptiveRadioCompressor

    compressor = AdaptiveRadioCompressor(
        input_dim=8,
        output_dim=4,
        hidden_dim=16,
        sample_pixels=8,
        min_spatial_std=0.2,
        std_weight=0.5,
        decorrelation_weight=0.25,
    )
    raw = torch.randn(2, 8, 4, 4)
    collapsed = torch.ones(2, 4, 4, 4) * 0.5
    losses = compressor.reconstruction_loss(raw, collapsed)

    assert {"std", "decorrelation"}.issubset(losses)
    assert losses["std"].item() > 0
    assert losses["decorrelation"].item() > 0
    assert losses["total"].requires_grad


def test_adaptive_compressor_can_highpass_raw_features_before_projection():
    from feature_field.dcff.radio_teacher import AdaptiveRadioCompressor

    compressor = AdaptiveRadioCompressor(
        input_dim=4,
        output_dim=2,
        hidden_dim=8,
        raw_highpass_kernel=3,
        normalize_output=False,
    )
    raw = torch.ones(1, 4, 5, 5)
    projected = compressor(raw)

    assert torch.allclose(projected, torch.zeros_like(projected), atol=1e-6)


def test_adaptive_compressor_highpass_adapter_preserves_raw_pca_base_at_init():
    from feature_field.dcff.radio_teacher import AdaptiveRadioCompressor

    components = torch.eye(4)
    mean = torch.zeros(4)
    compressor = AdaptiveRadioCompressor(
        input_dim=4,
        output_dim=2,
        hidden_dim=8,
        pca_state={"components": components, "mean": mean},
        adapter_highpass_kernel=3,
        normalize_output=False,
    )

    raw = torch.randn(1, 4, 3, 3)
    projected = compressor(raw)

    assert torch.allclose(projected, raw[:, :2], atol=1e-6)


def test_dcff_loss_accepts_asymmetric_fine_and_coarse_dims():
    loss_fn = DCFFLoss(lambda_tv=0.0)
    render = {
        "rgb": torch.rand(1, 3, 8, 8),
        "alpha": torch.ones(1, 1, 8, 8),
        "fine_features": F.normalize(torch.rand(1, 96, 4, 4), dim=1),
        "coarse_features": F.normalize(torch.rand(1, 32, 2, 2), dim=1),
    }
    gt = torch.rand(1, 3, 8, 8)
    fine_target = F.normalize(torch.rand(1, 96, 4, 4), dim=1)
    coarse_target = F.normalize(torch.rand(1, 32, 2, 2), dim=1)

    losses = loss_fn.compute(
        render_result=render,
        gt_rgb=gt,
        radio_geo=fine_target,
        radio_sem=coarse_target,
        phase=3,
    )

    assert torch.isfinite(losses["total"])
    assert "fine_total" in losses
    assert "coarse_total" in losses


def test_dcff_infonce_accepts_mixed_amp_and_cached_teacher_dtypes():
    from feature_field.dcff.losses import infonce_contrastive_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = F.normalize(torch.rand(1, 8, 8, 8, device=device).half(), dim=1)
    target = F.normalize(torch.rand(1, 8, 8, 8, device=device).float(), dim=1)
    mask = torch.ones(1, 1, 8, 8, device=device)

    loss = infonce_contrastive_loss(pred, target, mask=mask, n_samples=8)

    assert torch.isfinite(loss)
    assert loss.dtype == torch.float32


def test_dcff_loss_can_penalize_missing_feature_gradients():
    target = torch.zeros(1, 4, 6, 6)
    target[:, :, :, 3:] = 1.0
    smooth_pred = torch.zeros_like(target)

    loss_fn = DCFFLoss(
        lambda_tv=0.0,
        lambda_fine_cos=0.0,
        lambda_fine_l1=0.0,
        lambda_coarse_cos=0.0,
        lambda_coarse_l1=0.0,
        lambda_fine_grad=1.0,
        lambda_coarse_grad=1.0,
    )

    fine = loss_fn.fine_loss(smooth_pred, target)
    coarse = loss_fn.coarse_loss(smooth_pred, target)

    assert fine["fine_grad"].item() > 0
    assert coarse["coarse_grad"].item() > 0
    assert torch.allclose(fine["fine_total"], fine["fine_grad"])
    assert torch.allclose(coarse["coarse_total"], coarse["coarse_grad"])


def test_dcff_loss_can_weight_fine_edges_more_than_flat_regions():
    from feature_field.dcff.losses import edge_weighted_cosine_loss

    target = torch.zeros(1, 4, 6, 6)
    target[:, 0, :, :3] = 1.0
    target[:, 1, :, 3:] = 1.0
    pred = target.clone()
    pred[:, :, :, 3] = 0.0
    pred[:, 0, :, 3] = 1.0

    weighted = edge_weighted_cosine_loss(pred, target, edge_strength=3.0)
    unweighted = DCFFLoss(lambda_tv=0.0).fine_loss(pred, target)["fine_cos"]

    assert torch.isfinite(weighted)
    assert weighted.item() > unweighted.item()


def test_residual_spatial_fine_decoder_outputs_expected_shape_and_gradients():
    from feature_field.dcff.deferred_renderer import DeferredCascadedRenderer

    class DummyHashGrid(torch.nn.Module):
        input_mode = "implicit_scale"

        def forward(self, positions, scales=None, valid_mask=None):
            return torch.zeros(positions.shape[0], 8, device=positions.device)

        def total_variation_loss(self):
            return torch.tensor(0.0)

    renderer = DeferredCascadedRenderer(
        hash_grid=DummyHashGrid(),
        latent_dim=12,
        fine_latent_dim=8,
        coarse_latent_dim=4,
        fine_feature_dim=16,
        coarse_feature_dim=8,
        fine_hidden_dim=24,
        fine_decoder_type="residual_spatial",
    )
    z = torch.randn(2, 8, 5, 7, requires_grad=True)

    out = renderer.fine_decoder(z)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == (2, 16, 5, 7)
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_online_radio_teacher_uses_snapped_radio_resolution_for_feature_grid():
    from feature_field.dcff.radio_teacher import OnlineRadioTeacher

    class Resolution:
        height = 384
        width = 640

    class FakeRadio:
        def get_nearest_supported_resolution(self, height, width):
            assert (height, width) == (360, 640)
            return Resolution()

    teacher = OnlineRadioTeacher.__new__(OnlineRadioTeacher)
    teacher.radio = FakeRadio()
    teacher.patch_size = 16

    teacher.set_image_size(360, 640)

    assert teacher._radio_input_size == (384, 640)
    assert teacher.feature_resolution == (24, 40)


def test_deferred_renderer_splits_fine_and_coarse_latent_dims():
    from feature_field.dcff.deferred_renderer import DeferredCascadedRenderer

    class DummyHash(torch.nn.Module):
        feature_dim = 32
        input_mode = "implicit_scale"

        def forward(self, positions, scales=None, valid_mask=None, **kwargs):
            return torch.zeros(positions.shape[0], self.feature_dim, device=positions.device)

        def total_variation_loss(self):
            return torch.tensor(0.0)

    renderer = DeferredCascadedRenderer(
        hash_grid=DummyHash(),
        latent_dim=24,
        fine_latent_dim=16,
        coarse_latent_dim=8,
        fine_feature_dim=96,
        coarse_feature_dim=32,
        fine_hidden_dim=32,
        coarse_mode="carrier_residual",
    )

    assert renderer.fine_latent_dim == 16
    assert renderer.coarse_latent_dim == 8
    assert renderer.fine_decoder.decoder[0].in_channels == 16
    assert renderer.coarse_carrier_fusion.carrier_proj[0].in_channels == 8


def test_deferred_renderer_legacy_hash_uses_split_coarse_latent_dim():
    from feature_field.dcff.deferred_renderer import DeferredCascadedRenderer

    class DummyLegacyHash(torch.nn.Module):
        feature_dim = 32
        input_mode = "legacy"

        def __init__(self):
            super().__init__()
            self.latent_shapes = []

        def forward(self, positions, latent=None, view_dirs=None, valid_mask=None, **kwargs):
            self.latent_shapes.append(tuple(latent.shape))
            return torch.zeros(positions.shape[0], self.feature_dim, device=positions.device)

        def total_variation_loss(self):
            return torch.tensor(0.0)

    hash_grid = DummyLegacyHash()
    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid,
        latent_dim=64,
        fine_latent_dim=40,
        coarse_latent_dim=24,
        fine_feature_dim=64,
        coarse_feature_dim=32,
        fine_hidden_dim=32,
        coarse_mode="implicit_only",
    )

    position_map = torch.rand(2, 5, 7, 3)
    alpha = torch.ones(2, 1, 5, 7)
    z_coarse = torch.rand(2, 24, 5, 7)
    viewmat = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)

    coarse = renderer.decode_coarse(position_map, alpha, z_map=z_coarse, viewmat=viewmat)

    assert tuple(coarse.shape) == (2, 32, 5, 7)
    assert hash_grid.latent_shapes[-1] == (2 * 5 * 7, 24)


def test_coarse_carrier_residual_has_nonzero_initial_gradients():
    from feature_field.dcff.deferred_renderer import CarrierResidualCoarseFusion

    fusion = CarrierResidualCoarseFusion(
        latent_dim=8,
        feature_dim=16,
        carrier_hidden_dim=12,
        gate_hidden_dim=10,
    )
    z_map = torch.randn(2, 8, 5, 7, requires_grad=True)
    implicit = torch.randn(2, 16, 5, 7, requires_grad=True)
    target = torch.randn(2, 16, 5, 7)

    fused, carrier, gate = fusion(z_map, implicit)
    loss = F.mse_loss(fused, target)
    loss.backward()

    carrier_grad = fusion.carrier_proj[-1].weight.grad
    gate_grad = fusion.residual_gate[-2].weight.grad

    assert carrier.abs().mean().item() > 0
    assert gate.abs().mean().item() > 0
    assert carrier_grad is not None and carrier_grad.abs().sum().item() > 0
    assert gate_grad is not None and gate_grad.abs().sum().item() > 0


def test_spatial_hash_grid_chunked_forward_matches_unchunked():
    from feature_field.dcff.hash_grid import SpatialHashGrid

    torch.manual_seed(7)
    base = SpatialHashGrid(
        scene_extent=5.0,
        feature_dim=6,
        input_mode="legacy",
        latent_dim=4,
        n_levels=4,
        mlp_hidden=12,
        mlp_layers=2,
        forward_chunk_size=0,
    )
    chunked = SpatialHashGrid(
        scene_extent=5.0,
        feature_dim=6,
        input_mode="legacy",
        latent_dim=4,
        n_levels=4,
        mlp_hidden=12,
        mlp_layers=2,
        forward_chunk_size=5,
    )
    chunked.load_state_dict(base.state_dict())

    positions = torch.randn(23, 3)
    latent = torch.randn(23, 4)
    view_dirs = torch.randn(23, 3)
    valid = torch.arange(23) % 3 != 0

    expected = base(positions, latent=latent, view_dirs=view_dirs, valid_mask=valid)
    actual = chunked(positions, latent=latent, view_dirs=view_dirs, valid_mask=valid)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_coarse_carrier_residual_chunked_forward_matches_unchunked():
    from feature_field.dcff.deferred_renderer import CarrierResidualCoarseFusion

    torch.manual_seed(11)
    base = CarrierResidualCoarseFusion(
        latent_dim=5,
        feature_dim=7,
        carrier_hidden_dim=9,
        gate_hidden_dim=6,
        forward_batch_chunk_size=0,
    )
    chunked = CarrierResidualCoarseFusion(
        latent_dim=5,
        feature_dim=7,
        carrier_hidden_dim=9,
        gate_hidden_dim=6,
        forward_batch_chunk_size=2,
    )
    chunked.load_state_dict(base.state_dict())

    z_map = torch.randn(5, 5, 4, 3)
    implicit = torch.randn(5, 7, 4, 3)

    expected = base(z_map, implicit)
    actual = chunked(z_map, implicit)

    assert torch.allclose(actual[0], expected[0], atol=1e-6)
    assert actual[1] is None
    assert actual[2] is None


def test_deferred_renderer_supports_spatial_direct_coarse_mode():
    from feature_field.dcff.deferred_renderer import DeferredCascadedRenderer

    class DummyHash(torch.nn.Module):
        feature_dim = 11
        input_mode = "legacy"

        def forward(self, *args, **kwargs):
            raise AssertionError("spatial_direct coarse mode should not query hash grid")

        def total_variation_loss(self):
            return torch.tensor(0.0)

    renderer = DeferredCascadedRenderer(
        hash_grid=DummyHash(),
        latent_dim=16,
        fine_latent_dim=10,
        coarse_latent_dim=6,
        fine_feature_dim=13,
        coarse_feature_dim=11,
        fine_hidden_dim=16,
        coarse_mode="spatial_direct",
        coarse_carrier_hidden_dim=16,
    )

    z_coarse = torch.randn(3, 6, 5, 4)
    coarse, carrier, gate = renderer.coarse_carrier_fusion(z_coarse, None)

    assert tuple(coarse.shape) == (3, 11, 5, 4)
    assert carrier is None
    assert gate is None


def test_deferred_renderer_supports_spatial_full_direct_coarse_mode():
    from feature_field.dcff.deferred_renderer import DeferredCascadedRenderer

    class DummyHash(torch.nn.Module):
        feature_dim = 11
        input_mode = "legacy"

        def total_variation_loss(self):
            return torch.tensor(0.0)

    renderer = DeferredCascadedRenderer(
        hash_grid=DummyHash(),
        latent_dim=16,
        fine_latent_dim=10,
        coarse_latent_dim=6,
        fine_feature_dim=13,
        coarse_feature_dim=11,
        fine_hidden_dim=16,
        coarse_mode="spatial_full_direct",
        coarse_carrier_hidden_dim=16,
    )

    z_full = torch.randn(3, 16, 5, 4)
    coarse, carrier, gate = renderer.coarse_carrier_fusion(z_full, None)

    assert renderer.coarse_direct_uses_full_latent
    assert tuple(coarse.shape) == (3, 11, 5, 4)
    assert carrier is None
    assert gate is None


if __name__ == "__main__":
    test_adaptive_compressor_pca_initialization_matches_centered_projection()
    test_adaptive_compressor_reconstruction_loss_is_differentiable()
    test_adaptive_compressor_reconstruction_loss_supports_chunking()
    test_adaptive_compressor_regularizes_low_variance_and_channel_correlation()
    test_adaptive_compressor_can_highpass_raw_features_before_projection()
    test_adaptive_compressor_highpass_adapter_preserves_raw_pca_base_at_init()
    test_dcff_loss_accepts_asymmetric_fine_and_coarse_dims()
    test_dcff_infonce_accepts_mixed_amp_and_cached_teacher_dtypes()
    test_dcff_loss_can_penalize_missing_feature_gradients()
    test_dcff_loss_can_weight_fine_edges_more_than_flat_regions()
    test_residual_spatial_fine_decoder_outputs_expected_shape_and_gradients()
    test_online_radio_teacher_uses_snapped_radio_resolution_for_feature_grid()
    test_deferred_renderer_splits_fine_and_coarse_latent_dims()
    test_deferred_renderer_legacy_hash_uses_split_coarse_latent_dim()
    test_coarse_carrier_residual_has_nonzero_initial_gradients()
    test_spatial_hash_grid_chunked_forward_matches_unchunked()
    test_coarse_carrier_residual_chunked_forward_matches_unchunked()
    test_deferred_renderer_supports_spatial_direct_coarse_mode()
    test_deferred_renderer_supports_spatial_full_direct_coarse_mode()
    print("adaptive_radio_compression tests passed")
