# POFD-DenseFlow Single-Render Proposal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a single-render dense-flow proposal source for POFD Stage4 real-init refinement.

**Architecture:** Add an isolated dense-flow proposal module that predicts rendered-centered flow/confidence from POFD query/render features and render geometry, then reuse the existing Stage4 robust pose update and diagnostics. Keep the existing pair-matcher Stage4 path as the baseline and add `denseflow` as an opt-in proposal source.

**Tech Stack:** PyTorch, existing `feature_extract/tools/train_nvs_pose_feature_adapter.py`, existing POFD adapter/render cache, existing Stage4 diagnostics, `pytest`.

---

## File Structure

- Create: `feature_extract/denseflow_proposal.py`
  - Owns dense-flow GT target generation, local correlation, dense-flow head, flow loss helpers.
- Modify: `feature_extract/tools/train_nvs_pose_feature_adapter.py`
  - Adds CLI/config flags, optional dense-flow module construction, training loss wiring, Stage4 dense-flow eval branch, metrics.
- Modify: `tests/test_nvs_pose_feature_adapter.py`
  - Adds regression tests for GT flow, dense-flow head shapes, zero-flow pose no-op, and Stage4 cache-mode eval wiring.
- Create: `feature_extract/configs/pofd_denseflow_mixed_10_25_50_shuf.yaml`
  - First training config for frozen-adapter dense-flow probe.
- Create: `feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_val128_eval.yaml`
  - Stage4 dense-flow val128 eval config.
- Create: `feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_full182_eval.yaml`
  - Stage4 dense-flow full182 eval config.
- Modify: `docs/superpowers/plans/2026-05-16-pofd-stage4-single-render-continuous-cpr.md`
  - Append dense-flow experiment results after runs complete.
- Modify: `docs/superpowers/plans/2026-05-16-iclp-pofd-goal-audit.md`
  - Update checklist after dense-flow validation.

---

### Task 1: Dense-Flow Geometry Targets

**Files:**
- Create: `feature_extract/denseflow_proposal.py`
- Modify: `tests/test_nvs_pose_feature_adapter.py`

- [ ] **Step 1: Write failing GT-flow identity test**

Append this test to `tests/test_nvs_pose_feature_adapter.py`:

```python
def test_denseflow_gt_flow_identity_pose_is_zero():
    from feature_extract.denseflow_proposal import denseflow_gt_flow_from_render_depth

    depth = torch.ones(1, 1, 4, 5)
    intr = torch.tensor([[10.0, 10.0, 2.0, 1.5]])
    pose = torch.eye(4).view(1, 4, 4)

    out = denseflow_gt_flow_from_render_depth(
        pose_init=pose,
        pose_gt=pose,
        render_depth=depth,
        intrinsics=intr,
    )

    assert out["flow"].shape == (1, 2, 4, 5)
    assert out["valid"].shape == (1, 1, 4, 5)
    assert torch.allclose(out["flow"], torch.zeros_like(out["flow"]), atol=1.0e-6)
    assert out["valid"].all()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_denseflow_gt_flow_identity_pose_is_zero -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'feature_extract.denseflow_proposal'`.

- [ ] **Step 3: Implement GT-flow helper**

Create `feature_extract/denseflow_proposal.py` with:

```python
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _intrinsics_components(intrinsics: torch.Tensor, batch_size: int, device, dtype):
    intr = intrinsics.to(device=device, dtype=dtype)
    if intr.ndim == 1:
        intr = intr.view(1, 4)
    if intr.ndim != 2 or intr.shape[1] != 4:
        raise ValueError("intrinsics must have shape (4,) or (B,4)")
    if intr.shape[0] == 1 and batch_size > 1:
        intr = intr.expand(batch_size, -1)
    if intr.shape[0] != batch_size:
        raise ValueError(f"intrinsics batch {intr.shape[0]} does not match B={batch_size}")
    return intr[:, 0], intr[:, 1], intr[:, 2], intr[:, 3]


def scale_intrinsics_for_hw(
    intrinsics: torch.Tensor,
    *,
    source_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    src_h, src_w = int(source_hw[0]), int(source_hw[1])
    dst_h, dst_w = int(target_hw[0]), int(target_hw[1])
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    scaled = intrinsics.clone()
    scaled[..., 0] = scaled[..., 0] * sx
    scaled[..., 1] = scaled[..., 1] * sy
    scaled[..., 2] = scaled[..., 2] * sx
    scaled[..., 3] = scaled[..., 3] * sy
    return scaled


def denseflow_gt_flow_from_render_depth(
    *,
    pose_init: torch.Tensor,
    pose_gt: torch.Tensor,
    render_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    target_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    if pose_init.ndim != 3 or pose_init.shape[-2:] != (4, 4):
        raise ValueError("pose_init must have shape (B,4,4)")
    if pose_gt.shape != pose_init.shape:
        raise ValueError("pose_gt must have shape (B,4,4)")
    depth = render_depth.float()
    if depth.ndim == 3:
        depth = depth[:, None]
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("render_depth must have shape (B,1,H,W) or (B,H,W)")
    if target_hw is not None and tuple(depth.shape[-2:]) != tuple(target_hw):
        depth = F.interpolate(depth, size=target_hw, mode="bilinear", align_corners=False)
    bsz, _c, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype
    fx, fy, cx, cy = _intrinsics_components(intrinsics, bsz, device, dtype)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xx = xx.view(1, 1, height, width).expand(bsz, -1, -1, -1)
    yy = yy.view(1, 1, height, width).expand(bsz, -1, -1, -1)
    z = depth.clamp(min=1.0e-6)
    x = (xx - cx.view(bsz, 1, 1, 1)) * z / fx.view(bsz, 1, 1, 1).clamp(min=1.0e-6)
    y = (yy - cy.view(bsz, 1, 1, 1)) * z / fy.view(bsz, 1, 1, 1).clamp(min=1.0e-6)
    cam_init = torch.cat([x, y, z, torch.ones_like(z)], dim=1).flatten(2)
    rel = torch.bmm(pose_gt.float(), torch.linalg.inv(pose_init.float())).to(device=device, dtype=dtype)
    cam_gt = torch.bmm(rel[:, :3], cam_init).view(bsz, 3, height, width)
    z_gt = cam_gt[:, 2:3]
    u_gt = fx.view(bsz, 1, 1, 1) * cam_gt[:, 0:1] / z_gt.clamp(min=1.0e-6) + cx.view(bsz, 1, 1, 1)
    v_gt = fy.view(bsz, 1, 1, 1) * cam_gt[:, 1:2] / z_gt.clamp(min=1.0e-6) + cy.view(bsz, 1, 1, 1)
    flow = torch.cat([u_gt - xx, v_gt - yy], dim=1)
    valid = (
        (depth > 0.05)
        & (z_gt > 0.05)
        & (u_gt >= -0.5)
        & (u_gt <= float(width) - 0.5)
        & (v_gt >= -0.5)
        & (v_gt <= float(height) - 0.5)
    )
    return {"flow": flow * valid.to(dtype=flow.dtype), "valid": valid}
```

- [ ] **Step 4: Run identity test**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_denseflow_gt_flow_identity_pose_is_zero -q
```

Expected: PASS.

- [ ] **Step 5: Add translated-pose GT-flow test**

Append:

```python
def test_denseflow_gt_flow_translation_matches_projection_direction():
    from feature_extract.denseflow_proposal import denseflow_gt_flow_from_render_depth

    depth = torch.ones(1, 1, 3, 3)
    intr = torch.tensor([[9.0, 9.0, 1.0, 1.0]])
    pose_init = torch.eye(4).view(1, 4, 4)
    pose_gt = torch.eye(4).view(1, 4, 4)
    pose_gt[:, 0, 3] = 0.1

    out = denseflow_gt_flow_from_render_depth(
        pose_init=pose_init,
        pose_gt=pose_gt,
        render_depth=depth,
        intrinsics=intr,
    )

    center_flow_x = out["flow"][0, 0, 1, 1]
    center_flow_y = out["flow"][0, 1, 1, 1]
    assert center_flow_x > 0.0
    assert torch.isclose(center_flow_y, torch.tensor(0.0), atol=1.0e-5)
```

- [ ] **Step 6: Run translated-pose test**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_denseflow_gt_flow_translation_matches_projection_direction -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add feature_extract/denseflow_proposal.py tests/test_nvs_pose_feature_adapter.py
git commit -m "feat: add denseflow geometry targets"
```

---

### Task 2: Dense-Flow Head

**Files:**
- Modify: `feature_extract/denseflow_proposal.py`
- Modify: `tests/test_nvs_pose_feature_adapter.py`

- [ ] **Step 1: Write failing dense-flow head shape test**

Append:

```python
def test_pofd_denseflow_head_outputs_flow_and_confidence_shapes():
    from feature_extract.denseflow_proposal import PofdDenseFlowHead

    head = PofdDenseFlowHead(channels=8, radius=2, hidden_dim=16, zero_init=True)
    query = torch.randn(2, 8, 5, 6)
    render = torch.randn(2, 8, 5, 6)
    depth = torch.ones(2, 1, 5, 6)
    intr = torch.tensor([[20.0, 20.0, 3.0, 2.0], [20.0, 20.0, 3.0, 2.0]])

    out = head(query, render, depth=depth, intrinsics=intr)

    assert out["flow"].shape == (2, 2, 5, 6)
    assert out["confidence"].shape == (2, 1, 5, 6)
    assert out["corr"].shape == (2, 25, 5, 6)
    assert torch.all((out["confidence"] >= 0.0) & (out["confidence"] <= 1.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_pofd_denseflow_head_outputs_flow_and_confidence_shapes -q
```

Expected: FAIL with `ImportError` or `AttributeError` for `PofdDenseFlowHead`.

- [ ] **Step 3: Add local correlation and dense-flow head**

Append to `feature_extract/denseflow_proposal.py`:

```python
class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(max(1, min(8, out_ch // 8)), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def dense_local_correlation(query: torch.Tensor, render: torch.Tensor, radius: int) -> torch.Tensor:
    if query.shape != render.shape:
        raise ValueError(f"query and render must have same shape, got {query.shape} and {render.shape}")
    radius = int(radius)
    query_n = F.normalize(query.float(), dim=1, eps=1.0e-6)
    render_n = F.normalize(render.float(), dim=1, eps=1.0e-6)
    padded = F.pad(query_n, [radius] * 4)
    rows = []
    _, _, height, width = render_n.shape
    for dy in range(-radius, radius + 1):
        y0 = dy + radius
        for dx in range(-radius, radius + 1):
            x0 = dx + radius
            sample = padded[:, :, y0 : y0 + height, x0 : x0 + width]
            rows.append((render_n * sample).sum(dim=1))
    return torch.stack(rows, dim=1)


class PofdDenseFlowHead(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        radius: int = 8,
        hidden_dim: int = 64,
        zero_init: bool = True,
        max_flow_px: float | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.radius = int(radius)
        self.max_flow_px = float(max_flow_px if max_flow_px is not None else radius)
        corr_ch = (2 * self.radius + 1) ** 2
        context_ch = 5
        self.predict = nn.Sequential(
            ConvNormAct(corr_ch + context_ch, int(hidden_dim)),
            ConvNormAct(int(hidden_dim), int(hidden_dim)),
            nn.Conv2d(int(hidden_dim), 3, 1),
        )
        if zero_init:
            nn.init.zeros_(self.predict[-1].weight)
            if self.predict[-1].bias is not None:
                nn.init.zeros_(self.predict[-1].bias)

    def _context(self, depth: torch.Tensor | None, corr: torch.Tensor) -> torch.Tensor:
        bsz, _corr_ch, height, width = corr.shape
        device, dtype = corr.device, corr.dtype
        if depth is None:
            depth_ch = torch.zeros(bsz, 1, height, width, device=device, dtype=dtype)
            inv_depth = torch.zeros_like(depth_ch)
            valid = torch.ones_like(depth_ch)
        else:
            depth_ch = depth.float()
            if depth_ch.ndim == 3:
                depth_ch = depth_ch[:, None]
            if depth_ch.shape[-2:] != (height, width):
                depth_ch = F.interpolate(depth_ch, (height, width), mode="bilinear", align_corners=False)
            valid = (depth_ch > 0.05).to(dtype=dtype)
            log_depth = torch.log(depth_ch.clamp(min=0.05))
            inv_depth = 1.0 / depth_ch.clamp(min=0.05)
            depth_ch = (log_depth - log_depth.mean(dim=(-2, -1), keepdim=True)) / log_depth.std(
                dim=(-2, -1), keepdim=True, unbiased=False
            ).clamp(min=1.0e-6)
            inv_depth = (inv_depth - inv_depth.mean(dim=(-2, -1), keepdim=True)) / inv_depth.std(
                dim=(-2, -1), keepdim=True, unbiased=False
            ).clamp(min=1.0e-6)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        xy = torch.stack([xx, yy], dim=0).view(1, 2, height, width).expand(bsz, -1, -1, -1)
        return torch.cat([depth_ch.to(dtype=dtype), inv_depth.to(dtype=dtype), xy, valid], dim=1)

    def forward(
        self,
        query: torch.Tensor,
        render: torch.Tensor,
        *,
        depth: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if query.ndim != 4 or render.ndim != 4:
            raise ValueError("query and render must have shape (B,C,H,W)")
        if query.shape[0] != render.shape[0] or query.shape[-2:] != render.shape[-2:]:
            raise ValueError("query and render must share batch and spatial dimensions")
        channels = min(int(query.shape[1]), int(render.shape[1]), self.channels)
        corr = dense_local_correlation(query[:, :channels], render[:, :channels], self.radius)
        raw = self.predict(torch.cat([corr, self._context(depth, corr)], dim=1))
        flow = torch.tanh(raw[:, :2]) * self.max_flow_px
        confidence = torch.sigmoid(raw[:, 2:3])
        return {"flow": flow.to(dtype=query.dtype), "confidence": confidence.to(dtype=query.dtype), "corr": corr}
```

- [ ] **Step 4: Run dense-flow head test**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_pofd_denseflow_head_outputs_flow_and_confidence_shapes -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add feature_extract/denseflow_proposal.py tests/test_nvs_pose_feature_adapter.py
git commit -m "feat: add pofd denseflow head"
```

---

### Task 3: Dense-Flow Pose Update Wrapper

**Files:**
- Modify: `feature_extract/tools/train_nvs_pose_feature_adapter.py`
- Modify: `tests/test_nvs_pose_feature_adapter.py`

- [ ] **Step 1: Write failing zero-flow no-op test**

Append:

```python
def test_denseflow_pose_update_zero_flow_keeps_pose():
    from feature_extract.tools.train_nvs_pose_feature_adapter import denseflow_pose_update_from_render

    pose = torch.eye(4).view(1, 4, 4)
    flow = torch.zeros(1, 2, 3, 3)
    confidence = torch.ones(1, 1, 3, 3)
    yy, xx = torch.meshgrid(torch.arange(3.0), torch.arange(3.0), indexing="ij")
    z = torch.ones_like(xx)
    position = torch.stack([xx, yy, z], dim=0).unsqueeze(0)
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    intr = torch.tensor([[10.0, 10.0, 1.0, 1.0]])

    out = denseflow_pose_update_from_render(
        flow=flow,
        confidence=confidence,
        render_position=position,
        render_mask=mask,
        intrinsics=intr,
        current_pose=pose,
        min_points=4,
        max_update_trans_m=0.25,
        max_update_rot_deg=8.0,
    )

    assert out["success"][0]
    assert torch.allclose(out["pose"], pose, atol=1.0e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_denseflow_pose_update_zero_flow_keeps_pose -q
```

Expected: FAIL with missing `denseflow_pose_update_from_render`.

- [ ] **Step 3: Implement wrapper using existing Stage4 solver**

Add imports near the top of `feature_extract/tools/train_nvs_pose_feature_adapter.py`:

```python
from feature_extract.denseflow_proposal import PofdDenseFlowHead, denseflow_gt_flow_from_render_depth
```

Add this helper near `robust_pose_update_from_correspondences(...)`:

```python
def denseflow_pose_update_from_render(
    *,
    flow: torch.Tensor,
    confidence: torch.Tensor,
    render_position: torch.Tensor,
    render_mask: torch.Tensor | None,
    intrinsics: torch.Tensor,
    current_pose: torch.Tensor,
    min_points: int,
    max_update_trans_m: float,
    max_update_rot_deg: float,
    iterations: int = 5,
    damping: float = 1.0e-3,
    huber_delta_px: float = 3.0,
) -> Dict[str, torch.Tensor]:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape (B,2,H,W)")
    if confidence.ndim == 3:
        confidence = confidence[:, None]
    if confidence.ndim != 4 or confidence.shape[1] != 1:
        raise ValueError("confidence must have shape (B,1,H,W) or (B,H,W)")
    if render_position.ndim != 4 or render_position.shape[1] != 3:
        raise ValueError("render_position must have shape (B,3,H,W)")
    bsz, _two, height, width = flow.shape
    position = render_position.float()
    if position.shape[-2:] != (height, width):
        position = F.interpolate(position, size=(height, width), mode="bilinear", align_corners=False)
    conf = confidence.float()
    if conf.shape[-2:] != (height, width):
        conf = F.interpolate(conf, size=(height, width), mode="bilinear", align_corners=False)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    base_xy = torch.stack([xx, yy], dim=-1).view(1, height, width, 2).expand(bsz, -1, -1, -1)
    query_xy = base_xy + flow.permute(0, 2, 3, 1).float()
    valid = torch.isfinite(position).all(dim=1) & torch.isfinite(query_xy).all(dim=-1)
    if render_mask is not None:
        mask = render_mask
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.shape[-2:] != (height, width):
            mask = F.interpolate(mask.float(), size=(height, width), mode="nearest") > 0.5
        valid = valid & mask[:, 0].bool()
    return robust_pose_update_from_correspondences(
        position.permute(0, 2, 3, 1).reshape(bsz, height * width, 3),
        query_xy.reshape(bsz, height * width, 2),
        intrinsics,
        current_pose,
        weights=conf[:, 0].reshape(bsz, height * width),
        valid_mask=valid.reshape(bsz, height * width),
        iterations=iterations,
        damping=damping,
        huber_delta_px=huber_delta_px,
        min_points=min_points,
        max_update_trans_m=max_update_trans_m,
        max_update_rot_deg=max_update_rot_deg,
    )
```

- [ ] **Step 4: Run zero-flow test**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_denseflow_pose_update_zero_flow_keeps_pose -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add feature_extract/tools/train_nvs_pose_feature_adapter.py tests/test_nvs_pose_feature_adapter.py
git commit -m "feat: add denseflow stage4 pose update wrapper"
```

---

### Task 4: Dense-Flow Losses And Training Metrics

**Files:**
- Modify: `feature_extract/denseflow_proposal.py`
- Modify: `feature_extract/tools/train_nvs_pose_feature_adapter.py`
- Modify: `tests/test_nvs_pose_feature_adapter.py`

- [ ] **Step 1: Write failing dense-flow loss test**

Append:

```python
def test_denseflow_losses_zero_for_perfect_flow_on_valid_pixels():
    from feature_extract.denseflow_proposal import denseflow_supervision_loss

    pred_flow = torch.tensor([[[[1.0, 0.0]], [[0.0, -1.0]]]])
    gt_flow = pred_flow.clone()
    valid = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    conf = torch.ones(1, 1, 1, 2)

    loss, metrics = denseflow_supervision_loss(
        pred_flow=pred_flow,
        pred_confidence=conf,
        gt_flow=gt_flow,
        valid=valid,
    )

    assert loss.item() < 1.0e-6
    assert metrics["denseflow_flow_epe_px"].item() < 1.0e-6
    assert metrics["denseflow_valid_frac"].item() == 1.0
```

- [ ] **Step 2: Run loss test to verify it fails**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_denseflow_losses_zero_for_perfect_flow_on_valid_pixels -q
```

Expected: FAIL with missing `denseflow_supervision_loss`.

- [ ] **Step 3: Implement dense-flow loss helper**

Append to `feature_extract/denseflow_proposal.py`:

```python
def denseflow_supervision_loss(
    *,
    pred_flow: torch.Tensor,
    pred_confidence: torch.Tensor,
    gt_flow: torch.Tensor,
    valid: torch.Tensor,
    flow_weight: float = 1.0,
    confidence_weight: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if pred_flow.shape != gt_flow.shape:
        raise ValueError(f"pred_flow and gt_flow shape mismatch: {pred_flow.shape} vs {gt_flow.shape}")
    if valid.ndim == 3:
        valid = valid[:, None]
    valid_f = valid.to(device=pred_flow.device, dtype=pred_flow.dtype)
    if valid_f.shape[-2:] != pred_flow.shape[-2:]:
        valid_f = F.interpolate(valid_f, size=pred_flow.shape[-2:], mode="nearest")
    conf = pred_confidence
    if conf.ndim == 3:
        conf = conf[:, None]
    if conf.shape[-2:] != pred_flow.shape[-2:]:
        conf = F.interpolate(conf.float(), size=pred_flow.shape[-2:], mode="bilinear", align_corners=False)
    epe_map = torch.linalg.vector_norm(pred_flow.float() - gt_flow.float(), dim=1, keepdim=True)
    denom = valid_f.sum().clamp(min=1.0)
    flow_loss = F.smooth_l1_loss(pred_flow.float() * valid_f, gt_flow.float() * valid_f, reduction="sum") / denom
    conf_target = (epe_map.detach() < 1.0).to(dtype=conf.dtype) * valid_f
    conf_loss = F.binary_cross_entropy(conf.clamp(1.0e-4, 1.0 - 1.0e-4), conf_target, reduction="none")
    conf_loss = (conf_loss * valid_f).sum() / denom
    loss = float(flow_weight) * flow_loss + float(confidence_weight) * conf_loss
    metrics = {
        "denseflow_loss": loss.detach(),
        "denseflow_flow_loss": flow_loss.detach(),
        "denseflow_conf_loss": conf_loss.detach(),
        "denseflow_flow_epe_px": ((epe_map * valid_f).sum() / denom).detach(),
        "denseflow_valid_frac": valid_f.mean().detach(),
        "denseflow_conf_mean": ((conf * valid_f).sum() / denom).detach(),
    }
    return loss, metrics
```

- [ ] **Step 4: Run loss test**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_denseflow_losses_zero_for_perfect_flow_on_valid_pixels -q
```

Expected: PASS.

- [ ] **Step 5: Add CLI/config defaults**

In `parse_args()`, add:

```python
parser.add_argument("--denseflow-enabled", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--denseflow-weight", type=float, default=None)
parser.add_argument("--denseflow-flow-weight", type=float, default=None)
parser.add_argument("--denseflow-confidence-weight", type=float, default=None)
parser.add_argument("--denseflow-hidden-dim", type=int, default=None)
parser.add_argument("--denseflow-radius", type=int, default=None)
parser.add_argument("--denseflow-max-flow-px", type=float, default=None)
```

In `default_cfg`, add:

```python
"denseflow_enabled": False,
"denseflow_weight": 0.0,
"denseflow_flow_weight": 1.0,
"denseflow_confidence_weight": 0.0,
"denseflow_hidden_dim": 64,
"denseflow_radius": 8,
"denseflow_max_flow_px": 8.0,
```

- [ ] **Step 6: Instantiate dense-flow head and pass it to training**

After pair-matcher construction in `main()`, add:

```python
denseflow_head = None
if bool(args.denseflow_enabled):
    denseflow_head = PofdDenseFlowHead(
        channels=int(args.pose_feature_adapter_out_dim),
        radius=int(args.denseflow_radius),
        hidden_dim=int(args.denseflow_hidden_dim),
        max_flow_px=float(args.denseflow_max_flow_px),
    ).to(device)
```

When building optimizer parameter groups, append the dense-flow head parameters:

```python
if denseflow_head is not None:
    trainable.extend([p for p in denseflow_head.parameters() if p.requires_grad])
```

Change the `forward_batch(...)` signature to include:

```python
denseflow_head: PofdDenseFlowHead | None,
```

Update each `forward_batch(...)` call site to pass `denseflow_head`.

- [ ] **Step 7: Wire training loss**

In `forward_batch(...)`, initialize zero metrics near existing flow metrics:

```python
denseflow_loss = query_loc.new_zeros(())
denseflow_metrics = {
    "denseflow_loss": denseflow_loss.detach(),
    "denseflow_flow_loss": denseflow_loss.detach(),
    "denseflow_conf_loss": denseflow_loss.detach(),
    "denseflow_flow_epe_px": denseflow_loss.detach(),
    "denseflow_valid_frac": denseflow_loss.detach(),
    "denseflow_conf_mean": denseflow_loss.detach(),
}
```

After `target_intrinsics` is defined, add an opt-in one-render training branch:

```python
if denseflow_head is not None and float(args.denseflow_weight) > 0.0:
    with torch.no_grad():
        dense_batch = map_renderer.attach_pose_candidate_renders(
            dict(batch),
            init_pose[:, None].detach(),
            prefix="denseflow",
            require_grad=False,
            feature="all",
            include_aux=True,
        )
    dense_render_base = _single_candidate_tensor(dense_batch["denseflow_fine"], name="denseflow_fine").float()
    dense_depth = _single_candidate_tensor(dense_batch["denseflow_depth"], name="denseflow_depth").float()
    dense_mask = _single_candidate_tensor(dense_batch.get("denseflow_mask"), name="denseflow_mask")
    dense_intr = _single_candidate_tensor(dense_batch["denseflow_intrinsics"], name="denseflow_intrinsics").float()
    dense_render_loc = adapter.project_render(dense_render_base)
    dense_out = denseflow_head(query_loc, dense_render_loc, depth=dense_depth, intrinsics=dense_intr)
    gt = denseflow_gt_flow_from_render_depth(
        pose_init=init_pose.float(),
        pose_gt=pose_gt.float(),
        render_depth=dense_depth,
        intrinsics=dense_intr,
        target_hw=tuple(dense_out["flow"].shape[-2:]),
    )
    denseflow_loss, denseflow_metrics = denseflow_supervision_loss(
        pred_flow=dense_out["flow"],
        pred_confidence=dense_out["confidence"],
        gt_flow=gt["flow"],
        valid=gt["valid"] & (_resize_mask(dense_mask, tuple(gt["valid"].shape[-2:])) > 0.5),
        flow_weight=float(args.denseflow_flow_weight),
        confidence_weight=float(args.denseflow_confidence_weight),
    )
```

Add `+ float(args.denseflow_weight) * denseflow_loss` to the total `loss`.

Add `denseflow_metrics` to the final metrics dictionary with:

```python
metrics.update(denseflow_metrics)
```

- [ ] **Step 8: Run focused tests**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_denseflow_losses_zero_for_perfect_flow_on_valid_pixels tests/test_nvs_pose_feature_adapter.py::test_pofd_denseflow_head_outputs_flow_and_confidence_shapes -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add feature_extract/denseflow_proposal.py feature_extract/tools/train_nvs_pose_feature_adapter.py tests/test_nvs_pose_feature_adapter.py
git commit -m "feat: train denseflow proposal head"
```

---

### Task 5: Stage4 Dense-Flow Evaluation Path

**Files:**
- Modify: `feature_extract/tools/train_nvs_pose_feature_adapter.py`
- Modify: `tests/test_nvs_pose_feature_adapter.py`

- [ ] **Step 1: Write failing parser/eval-source test**

Append:

```python
def test_stage4_match_source_accepts_denseflow():
    source = Path("feature_extract/tools/train_nvs_pose_feature_adapter.py").read_text()
    assert '"denseflow"' in source
    assert "denseflow_pose_update_from_render" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_stage4_match_source_accepts_denseflow -q
```

Expected: FAIL until the parser/eval branch includes denseflow.

- [ ] **Step 3: Extend `--stage4-match-source` choices**

Change the parser line to:

```python
parser.add_argument("--stage4-match-source", choices=("pair_matcher", "local_corr", "denseflow"), default=None)
```

- [ ] **Step 4: Update eval function signature**

Change:

```python
def evaluate_stage4_single_render(..., pair_matcher: PairConditionedLocalMatcher, ...)
```

to include:

```python
denseflow_head: PofdDenseFlowHead | None,
```

Update every call site to pass `denseflow_head`.

- [ ] **Step 5: Add dense-flow eval branch**

Inside `evaluate_stage4_single_render(...)`, before the current pair-matcher
correspondence branch, add:

```python
if match_source == "denseflow":
    if denseflow_head is None:
        raise ValueError("stage4_match_source=denseflow requires --denseflow-enabled")
    denseflow_head.eval()
    dense_out = denseflow_head(query_loc, render_loc, depth=render_batch.get("stage4_depth"), intrinsics=intrinsics)
    last_solver = denseflow_pose_update_from_render(
        flow=dense_out["flow"],
        confidence=dense_out["confidence"],
        render_position=render_position,
        render_mask=render_mask,
        intrinsics=solver_intrinsics,
        current_pose=pose_cur,
        min_points=int(args.stage4_min_points),
        max_update_trans_m=float(args.stage4_max_update_trans_m),
        max_update_rot_deg=float(args.stage4_max_update_rot_deg),
        iterations=int(args.stage4_solver_iterations),
        damping=float(args.stage4_solver_damping),
        huber_delta_px=float(args.stage4_huber_delta_px),
    )
    solver_valid = last_solver["success"]
    match_valid_by_sample = solver_valid.float()
    match_conf_by_sample = dense_out["confidence"].flatten(1).mean(dim=1)
    match_offset_by_sample = torch.linalg.vector_norm(dense_out["flow"].float(), dim=1).flatten(1).mean(dim=1)
else:
    # keep existing local_corr / pair_matcher branches unchanged
```

Keep existing accept-gate, proposal virtual scoring, diagnostics, and dump code
after `last_solver` is created.

- [ ] **Step 6: Run parser/eval-source test**

Run:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_stage4_match_source_accepts_denseflow -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add feature_extract/tools/train_nvs_pose_feature_adapter.py tests/test_nvs_pose_feature_adapter.py
git commit -m "feat: evaluate denseflow stage4 updates"
```

---

### Task 6: Configs

**Files:**
- Create: `feature_extract/configs/pofd_denseflow_mixed_10_25_50_shuf.yaml`
- Create: `feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_val128_eval.yaml`
- Create: `feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_full182_eval.yaml`

- [ ] **Step 1: Create training config**

Create `feature_extract/configs/pofd_denseflow_mixed_10_25_50_shuf.yaml`:

```yaml
base_config: /root/ICLPose/feature_extract/configs/pofd_stage4_pairflow_mixed_10_25_50_shuf.yaml
exp_name: pofd_denseflow_mixed_10_25_50_shuf

training:
  batch_size: 2
  validation_batch_size: 2
  max_steps: 80
  eval_every: 20

nvs_pose_feature_adapter:
  denseflow_enabled: true
  denseflow_weight: 1.0
  denseflow_flow_weight: 1.0
  denseflow_confidence_weight: 0.05
  denseflow_hidden_dim: 64
  denseflow_radius: 8
  denseflow_max_flow_px: 8.0
  stage4_pair_match_flow_weight: 0.0
  pair_matcher_weight: 0.0
```

- [ ] **Step 2: Create val128 eval config**

Create `feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_val128_eval.yaml`:

```yaml
base_config: /root/ICLPose/feature_extract/configs/pofd_stage4_single_render_realinit_netvlad_renderloftr_top50_eval.yaml
exp_name: pofd_denseflow_realinit_renderloftr_top50_val128_eval

nvs_pose_feature_adapter:
  denseflow_enabled: true
  stage4_match_source: denseflow
  stage4_iterations: 1
  stage4_max_update_trans_m: 0.25
  stage4_max_update_rot_deg: 8.0
  stage4_eval_dump_path: stage4_denseflow_update_dump.jsonl
  stage4_eval_dump_max_rows: 512
```

- [ ] **Step 3: Create full182 eval config**

Create `feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_full182_eval.yaml`:

```yaml
base_config: /root/ICLPose/feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_val128_eval.yaml
exp_name: pofd_denseflow_realinit_renderloftr_top50_full182_eval

dataset:
  max_val_samples:
```

- [ ] **Step 4: Validate YAML parse through existing config loader**

Run:

```bash
python -m py_compile feature_extract/tools/train_nvs_pose_feature_adapter.py
```

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add feature_extract/configs/pofd_denseflow_mixed_10_25_50_shuf.yaml feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_val128_eval.yaml feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_full182_eval.yaml
git commit -m "config: add pofd denseflow probes"
```

---

### Task 7: Targeted Regression

**Files:**
- Modify only if tests reveal a real defect in files changed by Tasks 1-6.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_feature_extract_checkpoint_io.py tests/test_nvs_pose_feature_adapter.py -q
```

Expected: PASS with all tests passing.

- [ ] **Step 2: Run compile check**

Run:

```bash
python -m py_compile feature_extract/tools/train_nvs_pose_feature_adapter.py feature_extract/denseflow_proposal.py
```

Expected: exit 0.

- [ ] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit 0.

- [ ] **Step 4: Commit fixes if Step 1-3 required changes**

```bash
git add feature_extract/denseflow_proposal.py feature_extract/tools/train_nvs_pose_feature_adapter.py tests/test_nvs_pose_feature_adapter.py
git commit -m "test: stabilize denseflow proposal path"
```

---

### Task 8: GPU Smoke Experiments

**Files:**
- Modify: `docs/superpowers/plans/2026-05-16-pofd-stage4-single-render-continuous-cpr.md`
- Modify: `docs/superpowers/plans/2026-05-16-iclp-pofd-goal-audit.md`

- [ ] **Step 1: Check GPU availability**

Run:

```bash
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits
```

Expected: empty output or enough free GPUs for a smoke.

- [ ] **Step 2: Train F1 smoke**

Run with the repository's established training invocation for this script:

```bash
CUDA_VISIBLE_DEVICES=0 python feature_extract/tools/train_nvs_pose_feature_adapter.py \
  --config feature_extract/configs/pofd_denseflow_mixed_10_25_50_shuf.yaml
```

Expected: a result directory under `result/result/feature_extract/` with denseflow metrics in logs.

- [ ] **Step 3: Evaluate F1 on val128**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python feature_extract/tools/train_nvs_pose_feature_adapter.py \
  --config feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_val128_eval.yaml \
  --stage4-eval-only
```

Expected: `stage4_eval_summary.json` with `stage4_cost_gain_m` materially above the current +6mm val128 ceiling before expanding experiments.

- [ ] **Step 4: Stop or expand based on promote/continue gate**

If val128 gain is <=5mm or success@5cm/10cm/50cm regresses, stop the branch and record no-go.

If val128 gain is clearly above +6mm without success regression, run full182:

```bash
CUDA_VISIBLE_DEVICES=0 python feature_extract/tools/train_nvs_pose_feature_adapter.py \
  --config feature_extract/configs/pofd_denseflow_realinit_renderloftr_top50_full182_eval.yaml \
  --stage4-eval-only
```

- [ ] **Step 5: Update audit docs**

Generate a concise table from the eval JSON:

```bash
python - <<'PY'
import json
from pathlib import Path

summary_path = Path("result/result/feature_extract/pofd_denseflow_realinit_renderloftr_top50_val128_eval/stage4_eval_summary.json")
data = json.loads(summary_path.read_text())
m = data["metrics"]
status = "promote" if (
    m["stage4_cost_gain_m"] >= 0.074
    and m["stage4_pred_success_25cm_10deg"] >= 0.80
    and m["stage4_pred_success_5cm_2deg"] >= m["stage4_init_success_5cm_2deg"]
    and m["stage4_pred_success_10cm_5deg"] >= m["stage4_init_success_10cm_5deg"]
    and m["stage4_pred_success_50cm_10deg"] >= m["stage4_init_success_50cm_10deg"]
) else ("continue" if m["stage4_cost_gain_m"] > 0.006 else "stop")
print("POFD-DenseFlow Stage4 follow-up:")
print("")
print("| setting | init | final | gain | success@5cm/2deg | success@10cm/5deg | success@25cm/10deg | success@50cm/10deg | status |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
print(
    "| F1 val128 | "
    f"{m['stage4_init_cost_m']:.4f}m | "
    f"{m['stage4_pred_cost_m']:.4f}m | "
    f"{m['stage4_cost_gain_m'] * 1000.0:.2f}mm | "
    f"{m['stage4_init_success_5cm_2deg']:.3f} -> {m['stage4_pred_success_5cm_2deg']:.3f} | "
    f"{m['stage4_init_success_10cm_5deg']:.3f} -> {m['stage4_pred_success_10cm_5deg']:.3f} | "
    f"{m['stage4_init_success_25cm_10deg']:.3f} -> {m['stage4_pred_success_25cm_10deg']:.3f} | "
    f"{m['stage4_init_success_50cm_10deg']:.3f} -> {m['stage4_pred_success_50cm_10deg']:.3f} | "
    f"{status} |"
)
PY
```

Append the generated markdown to
`docs/superpowers/plans/2026-05-16-pofd-stage4-single-render-continuous-cpr.md`
and update `docs/superpowers/plans/2026-05-16-iclp-pofd-goal-audit.md` with the
same status decision.

- [ ] **Step 6: Commit experiment docs**

```bash
git add docs/superpowers/plans/2026-05-16-pofd-stage4-single-render-continuous-cpr.md docs/superpowers/plans/2026-05-16-iclp-pofd-goal-audit.md
git commit -m "docs: record pofd denseflow smoke results"
```

---

## Completion Criteria

The implementation is ready for a larger GPU matrix only when:

- Focused tests pass.
- `py_compile` passes.
- `git diff --check` passes.
- F1 val128 dense-flow eval beats the current Stage4 +6mm ceiling without 5cm/10cm/50cm success regression.

The active thread goal is not complete until the expert-file gates are met:

- q50 val128 / full182 real-init single-render improvement reaches the promote gates.
- q10/q25 regression checks pass.
- 3-seed stability passes.
- At least one additional scene/dataset is evaluated.
- SOTA/baseline comparison is reported with single-render, cache reranker, and multi-render baselines separated.
