from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np

from data.radio_loc_dataset import add_pose_noise
from pose_refine.models.concat_pose_net import (
    ConcatPoseNet,
    guided_local_correlation,
    local_correlation,
    soft_argmax_flow_from_correlation,
)
from pose_refine.runtime import (
    apply_pose_delta,
    build_concat_pose_model,
    load_local_matcher_weights,
)
from pose_refine.sparse_init import _solve_pnp
from pose_refine.tools.diag_dcff_cm_sensitivity import (
    correlation_wls_pose_update,
    load_query_local_matcher,
)
from pose_refine.train_impl import (
    ConcatLocTrainer,
    confidence_epe_bce_loss,
    compute_observability_flow_weight,
    feature_metric_pose_update_loss,
    has_trainable_localization_feature_path,
    local_correlation_subpixel_ce_loss_from_corr,
    local_correlation_soft_flow_loss_from_corr,
    masked_feature_cosine_distance_per_sample,
    flow_loss_fn,
    pose_loss,
    perturb_w2c_camera_center,
)
from pose_refine.utils.geometry_solver import (
    compute_image_jacobian,
    diff_pose_solve,
    feature_metric_solve,
)


def _offset_from_argmax(index: int, radius: int) -> tuple[int, int]:
    width = 2 * radius + 1
    return index // width - radius, index % width - radius


def test_render_centered_correlation_matches_render_to_query_flow():
    height, width = 7, 9
    channels = height * width
    rendered = torch.zeros(1, channels, height, width)
    query = torch.zeros_like(rendered)

    flow_dy, flow_dx = -1, 2
    for y in range(height):
        for x in range(width):
            c = y * width + x
            rendered[0, c, y, x] = 1.0
            yq, xq = y + flow_dy, x + flow_dx
            if 0 <= yq < height and 0 <= xq < width:
                query[0, c, yq, xq] = 1.0

    radius = 3
    corr = local_correlation(rendered, query, radius=radius)
    y, x = 3, 3
    best = int(corr[0, :, y, x].argmax())

    assert _offset_from_argmax(best, radius) == (flow_dy, flow_dx)


def test_guided_correlation_centers_offsets_on_same_pixel_flow():
    height, width = 5, 7
    rendered = torch.zeros(1, 1, height, width)
    query = torch.zeros_like(rendered)
    flow = torch.zeros(1, 2, height, width)

    y, x = 2, 2
    rendered[0, 0, y, x] = 1.0
    query[0, 0, y, x + 2] = 1.0
    flow[0, 0, y, x] = 1.0

    radius = 2
    corr = guided_local_correlation(rendered, query, flow, radius=radius)
    best = int(corr[0, :, y, x].argmax())

    assert _offset_from_argmax(best, radius) == (0, 1)


def test_checkpointed_guided_correlation_matches_default():
    torch.manual_seed(123)
    height, width = 5, 6
    rendered = F.normalize(torch.randn(1, 3, height, width), dim=1)
    query = F.normalize(torch.randn(1, 3, height, width), dim=1)
    flow = torch.randn(1, 2, height, width) * 0.25

    expected = guided_local_correlation(rendered, query, flow, radius=2)
    actual = guided_local_correlation(
        rendered,
        query,
        flow,
        radius=2,
        checkpoint_offsets=True,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_wls_recovers_render_to_query_flow_update():
    height, width = 16, 20
    v, u = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    depth = (4.0 + 0.6 * u + 0.4 * v + 0.2 * u * v).unsqueeze(0)
    intrinsics = {
        "fx": 80.0,
        "fy": 82.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }
    xi_true = torch.tensor([[0.03, -0.02, 0.01, 0.004, -0.003, 0.002]])

    Ju, Jv, valid = compute_image_jacobian(depth, intrinsics)
    flow_u = torch.bmm(Ju, xi_true.unsqueeze(-1)).squeeze(-1).reshape(1, height, width)
    flow_v = torch.bmm(Jv, xi_true.unsqueeze(-1)).squeeze(-1).reshape(1, height, width)
    flow = torch.stack([flow_u, flow_v], dim=1)
    confidence = torch.ones(1, 2, height, width)

    xi_pred = diff_pose_solve(flow, confidence, Ju, Jv, valid, damping=1e-6)

    assert torch.allclose(xi_pred, xi_true, atol=2e-4, rtol=2e-3)


def test_full_wls_can_keep_translation_and_use_hybrid_rotation_head():
    height, width = 8, 10
    model = ConcatPoseNet(
        feature_dim=4,
        hidden_dim=32,
        use_gru=False,
        full_wls=True,
        rot_mode="hybrid",
    )
    with torch.no_grad():
        model.rot_fc[-1].bias.copy_(torch.tensor([0.01, -0.02, 0.03]))

    depth = torch.ones(1, height, width) * 3.0
    flow = torch.zeros(1, 2, height, width)
    confidence = torch.ones(1, 1, height, width)
    intrinsics = {
        "fx": 40.0,
        "fy": 42.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }
    feat_for_trans = torch.ones(1, 16, height, width)

    delta_xi, delta_xi_full = model._solve_and_regress(
        flow,
        confidence,
        depth,
        intrinsics,
        feat_for_trans,
        query_coarse=None,
        irls_iters=0,
        robust_kernel="huber",
    )

    assert torch.allclose(delta_xi[:, :3], delta_xi_full[:, :3], atol=1e-6)
    assert torch.allclose(delta_xi[:, 3:], torch.tensor([[0.01, -0.02, 0.03]]), atol=1e-6)


def test_full_wls_hybrid_rotation_accepts_asymmetric_coarse_context():
    height, width = 8, 10
    model = ConcatPoseNet(
        feature_dim=4,
        coarse_feature_dim=2,
        hidden_dim=32,
        use_coarse=True,
        use_coarse_in_fine_head=True,
        use_gru=False,
        full_wls=True,
        rot_mode="hybrid",
    )

    depth = torch.ones(1, height, width) * 3.0
    flow = torch.zeros(1, 2, height, width)
    confidence = torch.ones(1, 1, height, width)
    intrinsics = {
        "fx": 40.0,
        "fy": 42.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }
    feat_for_trans = torch.ones(1, 16, height, width)
    query_coarse = torch.randn(1, 2, 3, 4)
    rendered_coarse = torch.randn(1, 2, 3, 4)

    delta_xi, delta_xi_full = model._solve_and_regress(
        flow,
        confidence,
        depth,
        intrinsics,
        feat_for_trans,
        query_coarse=query_coarse,
        rendered_coarse=rendered_coarse,
        irls_iters=0,
        robust_kernel="huber",
    )

    assert delta_xi.shape == (1, 6)
    assert torch.allclose(delta_xi[:, :3], delta_xi_full[:, :3], atol=1e-6)


def test_runtime_renders_map_coarse_for_fine_stage_context():
    class DummyModel(torch.nn.Module):
        use_coarse = True
        use_coarse_in_fine_head = True
        pose_update_scale = 1.0

        def should_run_coarse_stage(self, outer_iter=0):
            return False

        def forward_fine_stage(
            self,
            query_fine,
            rendered_fine,
            depth,
            intrinsics,
            irls_iters=None,
            robust_kernel=None,
            query_coarse=None,
            rendered_coarse=None,
        ):
            self.received_rendered_coarse = rendered_coarse
            return {
                "delta_xi": torch.zeros(query_fine.shape[0], 6),
                "flow": torch.zeros(query_fine.shape[0], 2, *query_fine.shape[-2:]),
                "confidence": torch.ones(query_fine.shape[0], 2, *query_fine.shape[-2:]),
            }

    from pose_refine.runtime import run_model_refine_iteration

    model = DummyModel()
    calls = []

    def render_batch_fn(_poses, render_coarse):
        calls.append(render_coarse)
        return {
            "fine_features": torch.zeros(1, 4, 5, 6),
            "coarse_features": torch.ones(1, 2, 3, 4) if render_coarse else None,
            "depth": torch.ones(1, 5, 6),
        }

    run_model_refine_iteration(
        model,
        render_batch_fn,
        query_fine=torch.zeros(1, 4, 5, 6),
        query_coarse=torch.zeros(1, 2, 3, 4),
        pose_cur=torch.eye(4).unsqueeze(0),
        render_intr={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )

    assert calls == [True]
    assert model.received_rendered_coarse is not None


def test_geodesic_pose_loss_keeps_gradient_below_one_degree():
    pose_init = torch.eye(4).unsqueeze(0)
    pose_gt = torch.eye(4).unsqueeze(0)
    angle = torch.tensor(0.1 * torch.pi / 180.0)
    pose_gt[0, 0, 0] = torch.cos(angle)
    pose_gt[0, 0, 1] = -torch.sin(angle)
    pose_gt[0, 1, 0] = torch.sin(angle)
    pose_gt[0, 1, 1] = torch.cos(angle)

    delta_xi = torch.zeros(1, 6, requires_grad=True)
    loss, _metrics = pose_loss(
        delta_xi,
        pose_init,
        pose_gt,
        rot_weight=1.0,
        trans_weight=0.0,
        rot_loss_type="geodesic",
        loss_mode="compose",
    )
    loss.backward()

    assert delta_xi.grad is not None
    assert delta_xi.grad[:, 3:].abs().max().item() > 1e-6


def test_corr_wls_diagnostic_applies_optional_depth_aware_matcher():
    class CenterMatcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_shape = None
            self.valid_shape = None

        def forward(self, corr, depth=None, valid_mask=None):
            self.depth_shape = None if depth is None else tuple(depth.shape)
            self.valid_shape = None if valid_mask is None else tuple(valid_mask.shape)
            refined = corr.new_full(corr.shape, -20.0)
            refined[:, corr.shape[1] // 2] = 20.0
            return refined

    height, width = 8, 10
    radius = 1
    query = torch.randn(1, 4, height, width)
    rendered = torch.randn(1, 4, height, width)
    depth = torch.ones(1, height, width) * 3.0
    pose = torch.eye(4).unsqueeze(0)
    intrinsics = {
        "fx": 40.0,
        "fy": 42.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }
    valid = torch.ones(1, 1, height, width)
    matcher = CenterMatcher()

    _delta, pose_pred, metrics = correlation_wls_pose_update(
        query,
        rendered,
        depth,
        pose,
        pose,
        intrinsics,
        radius=radius,
        temperature=0.05,
        damping=1e-3,
        valid_mask=valid,
        matcher=matcher,
    )

    assert matcher.depth_shape == (1, height, width)
    assert matcher.valid_shape == (1, 1, height, width)
    assert metrics["flow_epe"].item() < 1e-3
    assert torch.allclose(pose_pred, pose, atol=1e-4)


def test_corr_wls_diagnostic_loads_query_student_local_matcher_submodule():
    from feature_extract.students.radio_query_student import DepthAwareLocalMatcher

    matcher = DepthAwareLocalMatcher(radius=1, hidden_dim=12, residual_scale=0.25)
    with torch.no_grad():
        matcher.residual_scale.fill_(0.75)
        matcher.refine[-1].bias.fill_(0.125)
    checkpoint = {
        "model_state_dict": {
            f"local_matcher.{key}": value.detach().clone()
            for key, value in matcher.state_dict().items()
        }
    }

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        tmp_path = Path(tmp)
        config_path = tmp_path / "query.yaml"
        checkpoint_path = tmp_path / "query.pth"
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    "  local_matcher_enabled: true",
                    "  local_matcher_radius: 1",
                    "  local_matcher_hidden_dim: 12",
                    "  local_matcher_residual_scale: 0.25",
                ]
            ),
            encoding="utf-8",
        )
        torch.save(checkpoint, checkpoint_path)

        loaded = load_query_local_matcher(str(config_path), str(checkpoint_path), torch.device("cpu"))

    assert loaded is not None
    assert torch.allclose(loaded.residual_scale, torch.tensor(0.75))
    assert torch.allclose(loaded.refine[-1].bias, torch.full_like(loaded.refine[-1].bias, 0.125))


def test_image_jacobian_accepts_batched_intrinsics():
    depth = torch.ones(2, 4, 5)
    intrinsics = torch.tensor(
        [
            [10.0, 12.0, 2.0, 1.5],
            [20.0, 24.0, 2.0, 1.5],
        ]
    )

    Ju, Jv, valid = compute_image_jacobian(depth, intrinsics)

    assert Ju.shape == (2, 20, 6)
    assert Jv.shape == (2, 20, 6)
    assert valid.all()
    assert torch.isclose(Ju[1, 0, 0], 2.0 * Ju[0, 0, 0])


def test_feature_metric_step_has_forward_translation_sign():
    height, width = 18, 22
    depth = torch.ones(1, height, width)
    intrinsics = {
        "fx": 20.0,
        "fy": 20.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }

    v, u = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    rendered = torch.stack([u, v, u * u, v * v, u * v, torch.sin(2.0 * u)], dim=0).unsqueeze(0)

    tx_true = 0.02
    flow_px = intrinsics["fx"] * tx_true
    sample_x = (torch.arange(width).view(1, 1, width).expand(1, height, width) - flow_px)
    sample_y = torch.arange(height).view(1, height, 1).expand(1, height, width)
    grid_x = sample_x / (width - 1) * 2.0 - 1.0
    grid_y = sample_y / (height - 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    query = F.grid_sample(rendered, grid, mode="bilinear", padding_mode="border", align_corners=True)

    xi_pred, _ = feature_metric_solve(query, rendered, depth, intrinsics, damping=1e-3)

    assert xi_pred[0, 0] > 0.0
    assert torch.isclose(xi_pred[0, 0], torch.tensor(tx_true), atol=8e-3)


def test_pose_noise_rotation_preserves_camera_center():
    np.random.seed(7)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([12.0, -5.0, 3.0], dtype=np.float32)

    noisy = add_pose_noise(pose, rot_deg=10.0, trans_m=0.0)

    c_gt = -(pose[:3, :3].T @ pose[:3, 3])
    c_noisy = -(noisy[:3, :3].T @ noisy[:3, 3])
    assert np.linalg.norm(c_gt - c_noisy) < 1e-5


def test_camera_center_perturbation_preserves_rotation_and_metric_offset():
    pose = torch.eye(4).unsqueeze(0)
    yaw = torch.tensor(0.3)
    R = torch.tensor(
        [
            [torch.cos(yaw), -torch.sin(yaw), 0.0],
            [torch.sin(yaw), torch.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pose[0, :3, :3] = R
    pose[0, :3, 3] = torch.tensor([1.0, -2.0, 0.5])
    offset_cam = torch.tensor([[0.05, -0.02, 0.01]])

    perturbed = perturb_w2c_camera_center(pose, offset_cam, frame="camera")

    assert torch.allclose(perturbed[:, :3, :3], pose[:, :3, :3])
    c0 = -(pose[:, :3, :3].transpose(1, 2) @ pose[:, :3, 3:].contiguous()).squeeze(-1)
    c1 = -(perturbed[:, :3, :3].transpose(1, 2) @ perturbed[:, :3, 3:].contiguous()).squeeze(-1)
    expected_world = torch.bmm(R.T.unsqueeze(0), offset_cam.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(c1 - c0, expected_world, atol=1e-6)


def test_masked_feature_distance_prefers_aligned_render():
    query = torch.zeros(1, 4, 5, 6)
    query[:, 0] = 1.0
    aligned = query.clone()
    shifted = torch.roll(query, shifts=1, dims=3)
    shifted[:, 1, :, :2] = 1.0
    mask = torch.zeros(1, 1, 5, 6)
    mask[:, :, 1:4, 1:5] = 1.0

    pos = masked_feature_cosine_distance_per_sample(query, aligned, mask)
    neg = masked_feature_cosine_distance_per_sample(query, shifted, mask)

    assert pos.item() < 1e-6
    assert neg.item() > pos.item()


def test_local_correlation_soft_flow_loss_recovers_subpixel_offset():
    radius = 2
    window = 2 * radius + 1
    temperature = 0.2
    corr = torch.full((1, window * window, 1, 1), -40.0)

    def set_prob(dy: int, dx: int, prob: float) -> None:
        idx = (dy + radius) * window + (dx + radius)
        corr[0, idx, 0, 0] = torch.log(torch.tensor(prob)) * temperature

    set_prob(-1, 1, 0.375)
    set_prob(0, 1, 0.375)
    set_prob(-1, 2, 0.125)
    set_prob(0, 2, 0.125)

    flow_gt = torch.tensor([[[[1.25]], [[-0.5]]]])
    valid = torch.ones(1, 1, 1, 1)

    loss, metrics = local_correlation_soft_flow_loss_from_corr(
        corr,
        flow_gt,
        valid,
        radius=radius,
        temperature=temperature,
    )

    assert loss.item() < 1e-6
    assert metrics["corr_flow_epe"] < 1e-4


def test_flow_loss_fn_applies_normalized_weight_map():
    flow_pred = torch.tensor([[[[1.0, 3.0]], [[0.0, 0.0]]]])
    flow_gt = torch.zeros_like(flow_pred)
    valid = torch.ones(1, 1, 1, 2)
    weights = torch.tensor([[[[1.0, 3.0]]]])

    unweighted, _ = flow_loss_fn(flow_pred, flow_gt, valid, huber_delta=5.0)
    weighted, metrics = flow_loss_fn(
        flow_pred,
        flow_gt,
        valid,
        huber_delta=5.0,
        weight_map=weights,
    )

    assert torch.isclose(unweighted, torch.tensor(0.25), atol=1e-6)
    assert torch.isclose(weighted, torch.tensor(0.35), atol=1e-6)
    assert np.isclose(metrics["flow_weight_mean"], 2.0)


def test_observability_flow_weight_prioritizes_yaw_sensitive_edges():
    height, width = 5, 7
    depth = torch.ones(1, height, width) * 3.0
    valid = torch.ones(1, 1, height, width)
    intrinsics = {
        "fx": 30.0,
        "fy": 32.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }

    weights = compute_observability_flow_weight(
        depth,
        target_hw=(height, width),
        intrinsics=intrinsics,
        valid_mask=valid,
        mode="yaw",
        strength=1.0,
        max_weight=8.0,
    )

    center = weights[0, 0, height // 2, width // 2]
    corner = weights[0, 0, 0, 0]

    assert weights.shape == valid.shape
    assert corner > center
    assert torch.isclose((weights * valid).sum() / valid.sum(), torch.tensor(1.0), atol=1e-5)


def test_confidence_epe_bce_loss_prefers_high_confidence_on_accurate_flow():
    flow_gt = torch.zeros(1, 2, 1, 2)
    flow_pred = torch.tensor([[[[0.1, 6.0]], [[0.0, 0.0]]]])
    valid = torch.ones(1, 1, 1, 2)

    calibrated = torch.tensor([[[[0.9, 0.1]]]])
    inverted = torch.tensor([[[[0.1, 0.9]]]])

    good_loss, good_metrics = confidence_epe_bce_loss(
        flow_pred,
        flow_gt,
        calibrated,
        valid,
        good_px=1.0,
        bad_px=5.0,
    )
    bad_loss, _ = confidence_epe_bce_loss(
        flow_pred,
        flow_gt,
        inverted,
        valid,
        good_px=1.0,
        bad_px=5.0,
    )

    assert good_loss < bad_loss
    assert good_metrics["conf_epe_target_mean"] == 0.5
    assert good_metrics["conf_epe_pos_mean"] > good_metrics["conf_epe_neg_mean"]


def test_soft_argmax_flow_from_correlation_recovers_subpixel_offset():
    radius = 2
    window = 2 * radius + 1
    temperature = 0.2
    corr = torch.full((1, window * window, 1, 1), -40.0)

    def set_prob(dy: int, dx: int, prob: float) -> None:
        idx = (dy + radius) * window + (dx + radius)
        corr[0, idx, 0, 0] = torch.log(torch.tensor(prob)) * temperature

    set_prob(-1, 1, 0.375)
    set_prob(0, 1, 0.375)
    set_prob(-1, 2, 0.125)
    set_prob(0, 2, 0.125)

    flow = soft_argmax_flow_from_correlation(corr, radius=radius, temperature=temperature)

    assert torch.allclose(flow, torch.tensor([[[[1.25]], [[-0.5]]]]), atol=1e-4)


def test_corr_wls_mode_predicts_flow_from_local_match_peak():
    height, width = 4, 8
    channels = height * width
    rendered = torch.zeros(1, channels, height, width)
    query = torch.zeros_like(rendered)
    flow_dy, flow_dx = 1, -2
    for y in range(height):
        for x in range(width):
            c = y * width + x
            rendered[0, c, y, x] = 1.0
            yq, xq = y + flow_dy, x + flow_dx
            if 0 <= yq < height and 0 <= xq < width:
                query[0, c, yq, xq] = 1.0

    model = ConcatPoseNet(
        feature_dim=channels,
        use_corr_wls=True,
        local_radius=2,
        corr_wls_temperature=0.01,
        proj_mode="identity",
        full_wls=True,
    )
    ConcatPoseNet.BASE_INTRINSICS = {
        "fx": 50.0,
        "fy": 50.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }
    ConcatPoseNet.IMG_HW = (height, width)

    pred = model(
        query,
        rendered,
        torch.ones(1, height, width),
        ConcatPoseNet.BASE_INTRINSICS,
    )

    assert pred["flow"].shape == (1, 2, height, width)
    assert pred["confidence"].shape == (1, 2, height, width)
    expected_flow = torch.tensor([flow_dx, flow_dy], dtype=pred["flow"].dtype)
    assert torch.allclose(pred["flow"][0, :, 2, 3], expected_flow, atol=1e-3)
    assert pred["delta_xi"].shape == (1, 6)


def test_corr_wls_mode_applies_optional_depth_aware_matcher():
    class RightBiasMatcher(torch.nn.Module):
        def __init__(self, radius: int):
            super().__init__()
            self.radius = radius
            self.seen_depth_shape = None
            self.seen_valid_shape = None

        def forward(self, corr, depth=None, valid_mask=None):
            self.seen_depth_shape = None if depth is None else tuple(depth.shape)
            self.seen_valid_shape = None if valid_mask is None else tuple(valid_mask.shape)
            refined = corr.new_full(corr.shape, -20.0)
            window = 2 * self.radius + 1
            right_index = self.radius * window + (self.radius + 1)
            refined[:, right_index] = 20.0
            return refined

    height, width = 5, 6
    channels = 3
    query = torch.zeros(1, channels, height, width)
    rendered = torch.zeros_like(query)
    depth = torch.ones(1, height, width)
    radius = 1
    matcher = RightBiasMatcher(radius)

    model = ConcatPoseNet(
        feature_dim=channels,
        use_corr_wls=True,
        local_radius=radius,
        corr_wls_temperature=0.05,
        proj_mode="identity",
        full_wls=True,
    )
    model.local_matcher = matcher

    pred = model(
        query,
        rendered,
        depth,
        {
            "fx": 30.0,
            "fy": 30.0,
            "cx": (width - 1) / 2.0,
            "cy": (height - 1) / 2.0,
        },
    )

    assert matcher.seen_depth_shape == (1, height, width)
    assert matcher.seen_valid_shape == (1, 1, height, width)
    assert pred["flow"][:, 0].mean().item() > 0.95
    assert pred["flow"][:, 1].abs().mean().item() < 1e-4


def test_pose_runtime_builds_depth_aware_local_matcher_from_config():
    model = build_concat_pose_model(
        {
            "feature_dim": 4,
            "use_corr_wls": True,
            "full_wls": True,
            "local_radius": 2,
            "local_matcher_enabled": True,
            "local_matcher_hidden_dim": 12,
            "local_matcher_zero_init": True,
            "local_matcher_residual_scale": 0.25,
        },
        torch.device("cpu"),
    )

    assert model.local_matcher is not None
    assert model.local_matcher.radius == 2
    assert torch.allclose(model.local_matcher.residual_scale, torch.tensor(0.25))


def test_pose_runtime_depth_aware_local_matcher_zero_init_is_noop():
    model = build_concat_pose_model(
        {
            "feature_dim": 4,
            "use_corr_wls": True,
            "full_wls": True,
            "local_radius": 1,
            "local_matcher_enabled": True,
            "local_matcher_hidden_dim": 12,
            "local_matcher_zero_init": True,
        },
        torch.device("cpu"),
    )
    corr = torch.randn(1, 9, 4, 5)
    depth = torch.ones(1, 4, 5)

    refined = model.local_matcher(corr, depth=depth)

    assert torch.allclose(refined, corr, atol=1e-6)


def test_pose_runtime_builds_depth_aware_local_flow_head_from_config():
    model = build_concat_pose_model(
        {
            "feature_dim": 4,
            "use_gru": True,
            "full_wls": True,
            "local_radius": 2,
            "local_flow_head_enabled": True,
            "local_flow_head_hidden_dim": 12,
            "local_flow_head_zero_init": True,
            "local_flow_head_max_flow": 1.5,
            "local_flow_head_base_flow_mode": "argmax",
        },
        torch.device("cpu"),
    )

    assert model.local_flow_head is not None
    assert model.local_flow_head.radius == 2
    assert model.local_flow_head.max_flow == 1.5
    assert model.local_flow_head.base_flow_mode == "argmax"


def test_pose_runtime_builds_observability_aware_local_flow_head_from_config():
    model = build_concat_pose_model(
        {
            "feature_dim": 4,
            "use_gru": True,
            "full_wls": True,
            "local_radius": 1,
            "local_flow_head_enabled": True,
            "local_flow_head_hidden_dim": 12,
            "local_flow_head_zero_init": True,
            "local_flow_head_context_mode": "observability",
        },
        torch.device("cpu"),
    )

    assert model.local_flow_head is not None
    assert model.local_flow_head.context_mode == "observability"
    assert model.local_flow_head.predict[0].block[0].in_channels == 9 + 12


def test_gru_zero_iter_can_seed_flow_from_depth_aware_local_flow_head():
    height, width = 5, 7
    channels = height * width
    rendered = torch.zeros(1, channels, height, width)
    query = torch.zeros_like(rendered)
    for y in range(height):
        for x in range(width - 1):
            c = y * width + x
            rendered[0, c, y, x] = 1.0
            query[0, c, y, x + 1] = 1.0
    depth = torch.ones(1, height, width)

    model = ConcatPoseNet(
        feature_dim=channels,
        hidden_dim=32,
        use_gru=True,
        gru_iters=0,
        local_radius=1,
        proj_mode="identity",
        full_wls=True,
        local_flow_head_enabled=True,
        local_flow_head_zero_init=True,
        local_flow_head_base_flow_mode="argmax",
        local_flow_head_max_flow=1.0,
    )
    pred = model(
        query,
        rendered,
        depth,
        {
            "fx": 30.0,
            "fy": 30.0,
            "cx": (width - 1) / 2.0,
            "cy": (height - 1) / 2.0,
        },
    )

    assert pred["flow"][0, 0, 2, 2].item() > 0.9
    assert pred["flow"][0, 1, 2, 2].abs().item() < 1e-4
    assert torch.allclose(
        pred["confidence"][0, :, 2, 2],
        torch.full((2,), 0.5),
        atol=1e-6,
    )


def test_gru_zero_iter_accepts_observability_context_for_local_flow_head():
    height, width = 5, 7
    channels = height * width
    rendered = torch.zeros(1, channels, height, width)
    query = torch.zeros_like(rendered)
    for y in range(height):
        for x in range(width - 1):
            c = y * width + x
            rendered[0, c, y, x] = 1.0
            query[0, c, y, x + 1] = 1.0
    depth = torch.ones(1, height, width) * 3.0

    model = ConcatPoseNet(
        feature_dim=channels,
        hidden_dim=32,
        use_gru=True,
        gru_iters=0,
        local_radius=1,
        proj_mode="identity",
        full_wls=True,
        local_flow_head_enabled=True,
        local_flow_head_zero_init=True,
        local_flow_head_base_flow_mode="argmax",
        local_flow_head_context_mode="observability",
    )
    pred = model(
        query,
        rendered,
        depth,
        {
            "fx": 30.0,
            "fy": 32.0,
            "cx": (width - 1) / 2.0,
            "cy": (height - 1) / 2.0,
        },
    )

    assert pred["flow"].shape == (1, 2, height, width)
    assert pred["confidence"].shape == (1, 2, height, width)
    assert pred["flow"][0, 0, 2, 2].item() > 0.9


def test_gru_forward_exposes_local_flow_head_initial_prediction():
    height, width = 5, 7
    channels = height * width
    rendered = torch.zeros(1, channels, height, width)
    query = torch.zeros_like(rendered)
    for y in range(height):
        for x in range(width - 1):
            c = y * width + x
            rendered[0, c, y, x] = 1.0
            query[0, c, y, x + 1] = 1.0
    depth = torch.ones(1, height, width)

    model = ConcatPoseNet(
        feature_dim=channels,
        hidden_dim=32,
        use_gru=True,
        gru_iters=1,
        local_radius=1,
        proj_mode="identity",
        full_wls=True,
        local_flow_head_enabled=True,
        local_flow_head_zero_init=True,
        local_flow_head_base_flow_mode="argmax",
        local_flow_head_max_flow=1.0,
    )
    pred = model(
        query,
        rendered,
        depth,
        {
            "fx": 30.0,
            "fy": 30.0,
            "cx": (width - 1) / 2.0,
            "cy": (height - 1) / 2.0,
        },
    )

    assert "init_flow" in pred
    assert "init_confidence" in pred
    assert pred["init_flow"].shape == (1, 2, height, width)
    assert pred["init_confidence"].shape == (1, 1, height, width)
    assert pred["init_flow"][0, 0, 2, 2].item() > 0.9


def test_pose_runtime_loads_query_student_local_matcher_weights():
    source = build_concat_pose_model(
        {
            "feature_dim": 4,
            "use_corr_wls": True,
            "full_wls": True,
            "local_radius": 1,
            "local_matcher_enabled": True,
            "local_matcher_hidden_dim": 12,
        },
        torch.device("cpu"),
    )
    with torch.no_grad():
        source.local_matcher.residual_scale.fill_(0.75)
        source.local_matcher.refine[-1].bias.fill_(0.125)

    target = build_concat_pose_model(
        {
            "feature_dim": 4,
            "use_corr_wls": True,
            "full_wls": True,
            "local_radius": 1,
            "local_matcher_enabled": True,
            "local_matcher_hidden_dim": 12,
        },
        torch.device("cpu"),
    )

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        checkpoint_path = Path(tmp) / "query_student.pth"
        torch.save(
            {
                "model_state_dict": {
                    f"local_matcher.{key}": value.detach().clone()
                    for key, value in source.local_matcher.state_dict().items()
                }
            },
            checkpoint_path,
        )
        load_local_matcher_weights(target, str(checkpoint_path), torch.device("cpu"))

    assert torch.allclose(target.local_matcher.residual_scale, torch.tensor(0.75))
    assert torch.allclose(target.local_matcher.refine[-1].bias, torch.full_like(target.local_matcher.refine[-1].bias, 0.125))


def test_local_correlation_subpixel_ce_loss_recovers_soft_target_offset():
    radius = 2
    window = 2 * radius + 1
    temperature = 0.2
    corr = torch.full((1, window * window, 1, 1), -40.0)

    def set_prob(dy: int, dx: int, prob: float) -> None:
        idx = (dy + radius) * window + (dx + radius)
        corr[0, idx, 0, 0] = torch.log(torch.tensor(prob)) * temperature

    set_prob(-1, 1, 0.375)
    set_prob(0, 1, 0.375)
    set_prob(-1, 2, 0.125)
    set_prob(0, 2, 0.125)

    flow_gt = torch.tensor([[[[1.25]], [[-0.5]]]])
    valid = torch.ones(1, 1, 1, 1)

    loss, metrics = local_correlation_subpixel_ce_loss_from_corr(
        corr,
        flow_gt,
        valid,
        radius=radius,
        temperature=temperature,
    )

    assert torch.isfinite(loss)
    assert metrics["corr_subpx_flow_epe"] < 1e-4


def test_apply_pose_delta_zero_scale_keeps_pose_fixed():
    pose = torch.eye(4).unsqueeze(0)
    delta = torch.tensor([[0.1, -0.2, 0.3, 0.01, -0.02, 0.03]])

    updated = apply_pose_delta(pose, delta, scale=0.0)

    assert torch.allclose(updated, pose, atol=1e-7)


def test_apply_pose_delta_can_scale_translation_and_rotation_separately():
    pose = torch.eye(4).unsqueeze(0)
    trans_delta = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]])
    rot_delta = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])

    half_trans = apply_pose_delta(pose, trans_delta, trans_scale=0.5, rot_scale=1.0)
    no_rot = apply_pose_delta(pose, rot_delta, trans_scale=1.0, rot_scale=0.0)

    assert torch.allclose(half_trans[0, :3, 3], torch.tensor([0.1, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(no_rot, pose, atol=1e-6)


def test_runtime_uses_after_first_component_update_scales():
    class DummyModel(torch.nn.Module):
        pose_update_scale = 1.0
        pose_update_trans_scale = 1.0
        pose_update_rot_scale = 1.0
        pose_update_trans_scale_after_first = 0.25
        pose_update_rot_scale_after_first = 1.0

        def should_run_coarse_stage(self, outer_iter=0):
            return False

        def forward_fine_stage(
            self,
            query_fine,
            rendered_fine,
            depth,
            intrinsics,
            query_coarse=None,
            rendered_coarse=None,
            irls_iters=None,
            robust_kernel=None,
        ):
            return {
                "delta_xi": torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                "flow": torch.zeros(1, 2, *query_fine.shape[-2:]),
                "confidence": torch.ones(1, 2, *query_fine.shape[-2:]),
            }

    from pose_refine.runtime import run_model_refine_iteration

    model = DummyModel()

    def render_batch_fn(_poses, _render_coarse):
        return {
            "fine_features": torch.zeros(1, 4, 3, 4),
            "depth": torch.ones(1, 3, 4),
            "coarse_features": None,
        }

    state = run_model_refine_iteration(
        model,
        render_batch_fn,
        query_fine=torch.zeros(1, 4, 3, 4),
        query_coarse=None,
        pose_cur=torch.eye(4).unsqueeze(0),
        render_intr={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        outer_iter=1,
    )

    assert torch.allclose(state["pose_next"][0, :3, 3], torch.tensor([0.05, 0.0, 0.0]), atol=1e-6)


def test_projection_only_path_enables_localization_feature_losses():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_shared = torch.nn.Conv2d(4, 4, 1)

    model = DummyModel()

    assert has_trainable_localization_feature_path(
        model,
        loc_use_projection=True,
        map_decoder_active=False,
        map_fsm_active=False,
    )

    for param in model.proj_shared.parameters():
        param.requires_grad_(False)

    assert not has_trainable_localization_feature_path(
        model,
        loc_use_projection=True,
        map_decoder_active=False,
        map_fsm_active=False,
    )


def test_checkpoint_saves_runtime_dcff_state_when_decoder_is_frozen(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    trainer = object.__new__(ConcatLocTrainer)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer,
        lr_lambda=lambda _: 1.0,
    )
    trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
    trainer.best_val_trans = 0.123
    trainer.global_step = 7
    trainer.config = {"exp_name": "unit"}
    trainer.ckpt_dir = tmp_path
    trainer.save_epoch_checkpoints = False
    trainer.save_every = 0
    trainer.finetune_decoder = False
    trainer.finetune_fsm = False
    trainer.feat_select = None
    trainer.dcff_renderer = SimpleNamespace(
        fine_decoder=torch.nn.Conv2d(2, 3, 1),
        coarse_carrier_fusion=torch.nn.Conv2d(3, 4, 1),
    )
    trainer.feat_sharp_fine = torch.nn.Conv2d(3, 3, 1)

    trainer._save_checkpoint(epoch=0)

    ckpt = torch.load(tmp_path / "latest.pth", map_location="cpu")
    assert "fine_decoder_state" in ckpt
    assert "coarse_fusion_state" in ckpt
    assert "feat_sharp_state" in ckpt


def test_feature_metric_pose_update_loss_empty_mask_is_finite():
    height, width = 6, 8
    query = torch.randn(1, 3, height, width)
    rendered = torch.randn(1, 3, height, width)
    depth = torch.ones(1, height, width)
    pose = torch.eye(4).unsqueeze(0)
    intrinsics = {
        "fx": 20.0,
        "fy": 21.0,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
    }
    empty_mask = torch.zeros(1, 1, height, width)

    loss, metrics = feature_metric_pose_update_loss(
        query,
        rendered,
        depth,
        pose,
        pose,
        intrinsics,
        valid_mask=empty_mask,
    )

    assert torch.isfinite(loss)
    assert np.isfinite(metrics["fm_trans_err_mm"])


def test_solve_pnp_magsac_overload_returns_w2c_pose():
    import cv2

    if not hasattr(cv2, "UsacParams"):
        return

    pts_3d = []
    for z in [4.0, 5.0, 6.0]:
        for x in [-0.5, 0.0, 0.5]:
            for y in [-0.4, 0.2]:
                pts_3d.append([x, y, z])
    pts_3d = np.asarray(pts_3d, dtype=np.float64)
    intrinsics = {"fx": 300.0, "fy": 310.0, "cx": 160.0, "cy": 120.0}
    rvec_true = np.asarray([[0.03], [-0.02], [0.01]], dtype=np.float64)
    tvec_true = np.asarray([[0.1], [-0.05], [0.2]], dtype=np.float64)
    camera_matrix = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec_true, tvec_true, camera_matrix, None)
    pts_2d = pts_2d.reshape(-1, 2)

    pose, inliers = _solve_pnp(
        pts_3d,
        pts_2d,
        intrinsics,
        reproj_threshold=1.0,
        n_iters=1000,
        use_magsac=True,
    )

    R_true, _ = cv2.Rodrigues(rvec_true)
    expected = np.eye(4)
    expected[:3, :3] = R_true
    expected[:3, 3] = tvec_true[:, 0]
    assert inliers >= 12
    assert pose is not None
    assert np.allclose(pose, expected, atol=1e-4)


if __name__ == "__main__":
    test_render_centered_correlation_matches_render_to_query_flow()
    test_guided_correlation_centers_offsets_on_same_pixel_flow()
    test_wls_recovers_render_to_query_flow_update()
    test_image_jacobian_accepts_batched_intrinsics()
    test_feature_metric_step_has_forward_translation_sign()
    test_pose_noise_rotation_preserves_camera_center()
    test_camera_center_perturbation_preserves_rotation_and_metric_offset()
    test_masked_feature_distance_prefers_aligned_render()
    test_local_correlation_soft_flow_loss_recovers_subpixel_offset()
    test_soft_argmax_flow_from_correlation_recovers_subpixel_offset()
    test_corr_wls_mode_predicts_flow_from_local_match_peak()
    test_local_correlation_subpixel_ce_loss_recovers_soft_target_offset()
    test_apply_pose_delta_zero_scale_keeps_pose_fixed()
    test_projection_only_path_enables_localization_feature_losses()
    test_checkpoint_saves_runtime_dcff_state_when_decoder_is_frozen(Path("/tmp/iclpose_ckpt_test"))
    test_feature_metric_pose_update_loss_empty_mask_is_finite()
    test_solve_pnp_magsac_overload_returns_w2c_pose()
    print("pose refine geometry tests passed")
