from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from feature_extract.train_impl import augment_feature_with_scene_coord
from pose_refine.tools.eval_joint_corr_wls import (
    apply_confidence_filter,
    compute_query_map_flow,
    decode_local_correlation,
    make_initial_pose,
    maybe_project_local_corr_features,
    select_record_window,
)


def test_eval_decode_local_correlation_argmax_and_margin_confidence():
    corr = torch.zeros(1, 9, 2, 3)
    radius = 1
    dx, dy = 1, -1
    channel = (dy + radius) * (2 * radius + 1) + (dx + radius)
    corr[:, channel] = 4.0
    corr[:, 4] = 1.0

    flow, confidence, metrics = decode_local_correlation(
        corr,
        radius=radius,
        temperature=0.1,
        flow_decode_mode="argmax",
        confidence_mode="margin",
        confidence_margin_temperature=0.5,
    )

    assert torch.allclose(flow[:, 0], torch.ones_like(flow[:, 0]))
    assert torch.allclose(flow[:, 1], -torch.ones_like(flow[:, 1]))
    assert confidence.mean().item() > 0.99
    assert metrics["peak_margin"].item() > 2.9


def test_eval_decode_local_correlation_softargmax_keeps_subpixel_expectation():
    corr = torch.zeros(1, 9, 1, 1)
    radius = 1
    corr[:, 4] = 1.0
    corr[:, 5] = 1.0

    flow, confidence, metrics = decode_local_correlation(
        corr,
        radius=radius,
        temperature=1.0,
        flow_decode_mode="softargmax",
        confidence_mode="max",
    )

    assert 0.0 < flow[:, 0].item() < 0.5
    assert abs(flow[:, 1].item()) < 1e-6
    assert confidence.item() > 0.0
    assert "peak_top1" in metrics


def test_eval_apply_confidence_filter_keeps_top_fraction_per_image():
    confidence = torch.tensor([[[[0.1, 0.4], [0.8, 0.2]]]])
    valid = torch.ones_like(confidence)

    filtered, coverage = apply_confidence_filter(
        confidence,
        valid_mask=valid,
        threshold=0.0,
        top_fraction=0.5,
        power=1.0,
    )

    assert torch.count_nonzero(filtered).item() == 2
    assert filtered[0, 0, 1, 0].item() == confidence[0, 0, 1, 0].item()
    assert filtered[0, 0, 0, 1].item() == confidence[0, 0, 0, 1].item()
    assert coverage.item() == 0.5


def test_eval_select_record_window_applies_offset_before_limit():
    records = [{"id": idx} for idx in range(10)]

    selected = select_record_window(records, sample_offset=3, max_samples=4)

    assert [record["id"] for record in selected] == [3, 4, 5, 6]


def test_eval_make_initial_pose_is_stable_per_sample_name():
    pose = torch.eye(4).repeat(2, 1, 1)

    full = make_initial_pose(
        pose,
        noise_deg=1.0,
        noise_m=0.05,
        device=torch.device("cpu"),
        sample_names=["a.png", "b.png"],
        base_seed=123,
    )
    subset = make_initial_pose(
        pose[1:],
        noise_deg=1.0,
        noise_m=0.05,
        device=torch.device("cpu"),
        sample_names=["b.png"],
        base_seed=123,
    )

    assert torch.allclose(full[1:], subset)


def test_eval_scene_coord_augmented_correlation_recovers_local_shift():
    torch.manual_seed(7)
    height, width = 4, 5
    rendered_visual = torch.zeros(1, 1, height, width)
    query_visual = torch.zeros_like(rendered_visual)
    rendered_scene = torch.randn(1, 3, height, width)
    query_scene = torch.zeros_like(rendered_scene)
    query_scene[:, :, :, 1:] = rendered_scene[:, :, :, :-1]

    rendered_aug = augment_feature_with_scene_coord(rendered_visual, rendered_scene, weight=8.0)
    query_aug = augment_feature_with_scene_coord(query_visual, query_scene, weight=8.0)
    flow, confidence, metrics = compute_query_map_flow(
        rendered_aug,
        query_aug,
        radius=1,
        temperature=0.01,
        flow_decode_mode="argmax",
        confidence_mode="margin",
        confidence_margin_temperature=0.01,
    )

    assert torch.allclose(flow[:, 0, :, :-1], torch.ones_like(flow[:, 0, :, :-1]))
    assert torch.allclose(flow[:, 1, :, :-1], torch.zeros_like(flow[:, 1, :, :-1]))
    assert confidence[:, :, :, :-1].mean().item() > 0.9
    assert metrics["peak_margin"].item() > 0.0


def test_eval_local_corr_projector_can_project_map_only():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.local_corr_projector = torch.nn.Conv2d(2, 2, kernel_size=1, bias=False)
            with torch.no_grad():
                self.local_corr_projector.weight.copy_(2.0 * torch.eye(2).view(2, 2, 1, 1))

    rendered = torch.ones(1, 2, 3, 4)
    query = torch.full_like(rendered, 3.0)

    projected_rendered, projected_query = maybe_project_local_corr_features(
        _Model(),
        {"local_corr_projector_enabled": True, "local_corr_projector_apply_to_query": False},
        rendered,
        query,
    )

    assert torch.allclose(projected_rendered, rendered * 2.0)
    assert torch.allclose(projected_query, query)


def test_eval_local_corr_projector_projects_query_by_default():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.local_corr_projector = torch.nn.Conv2d(2, 2, kernel_size=1, bias=False)
            with torch.no_grad():
                self.local_corr_projector.weight.copy_(2.0 * torch.eye(2).view(2, 2, 1, 1))

    rendered = torch.ones(1, 2, 3, 4)
    query = torch.full_like(rendered, 3.0)

    projected_rendered, projected_query = maybe_project_local_corr_features(
        _Model(),
        {"local_corr_projector_enabled": True},
        rendered,
        query,
    )

    assert torch.allclose(projected_rendered, rendered * 2.0)
    assert torch.allclose(projected_query, query * 2.0)
