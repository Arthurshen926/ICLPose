from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np

from data.radio_loc_dataset import add_pose_noise
from pose_refine.models.concat_pose_net import guided_local_correlation, local_correlation
from pose_refine.runtime import apply_pose_delta
from pose_refine.sparse_init import _solve_pnp
from pose_refine.train_impl import (
    ConcatLocTrainer,
    feature_metric_pose_update_loss,
    has_trainable_localization_feature_path,
    local_correlation_subpixel_ce_loss_from_corr,
    local_correlation_soft_flow_loss_from_corr,
    masked_feature_cosine_distance_per_sample,
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
    test_feature_metric_step_has_forward_translation_sign()
    test_pose_noise_rotation_preserves_camera_center()
    test_camera_center_perturbation_preserves_rotation_and_metric_offset()
    test_masked_feature_distance_prefers_aligned_render()
    test_local_correlation_soft_flow_loss_recovers_subpixel_offset()
    test_local_correlation_subpixel_ce_loss_recovers_soft_target_offset()
    test_apply_pose_delta_zero_scale_keeps_pose_fixed()
    test_projection_only_path_enables_localization_feature_losses()
    test_checkpoint_saves_runtime_dcff_state_when_decoder_is_frozen(Path("/tmp/iclpose_ckpt_test"))
    test_feature_metric_pose_update_loss_empty_mask_is_finite()
    test_solve_pnp_magsac_overload_returns_w2c_pose()
    print("pose refine geometry tests passed")
