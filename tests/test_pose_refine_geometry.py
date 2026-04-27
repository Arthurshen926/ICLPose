from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from pose_refine.models.concat_pose_net import local_correlation
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


if __name__ == "__main__":
    test_render_centered_correlation_matches_render_to_query_flow()
    test_wls_recovers_render_to_query_flow_update()
    test_feature_metric_step_has_forward_translation_sign()
    print("pose refine geometry tests passed")
