#!/usr/bin/env python3
"""
MSFlowPoseNet 烟测试: 随机输入 → 全前向传播 → 检查输出形状.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ic_models.ms_flow_pose_net import MSFlowPoseNet

def smoke_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # v1 分辨率 (与现有特征匹配)
    coarse_hw = (7, 10)
    mid_hw = (15, 20)
    fine_hw = (35, 46)

    model = MSFlowPoseNet(
        hidden_dim=128, decode_dim=64,
        local_radius=4, damping=1e-3,
        coarse_hw=coarse_hw,
        mid_hw=mid_hw,
        fine_hw=fine_hw,
        fine_iters=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    B = 2
    query_feats = {
        'coarse':    torch.randn(B, 1280, *coarse_hw, device=device),
        'mid':       torch.randn(B, 1280, *mid_hw, device=device),
        'fine_sd':   torch.randn(B, 640, *fine_hw, device=device),   # v1: same as DINO
        'fine_dino': torch.randn(B, 768, *fine_hw, device=device),
    }
    render_feats = {
        'coarse':    torch.randn(B, 1280, *coarse_hw, device=device),
        'mid':       torch.randn(B, 1280, *mid_hw, device=device),
        'fine_sd':   torch.randn(B, 640, *fine_hw, device=device),
        'fine_dino': torch.randn(B, 768, *fine_hw, device=device),
    }
    depth = torch.rand(B, *fine_hw, device=device) * 3.0 + 0.5

    print("\nForward pass...")
    out = model(query_feats, render_feats, depth)

    print("\nOutputs:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}: {list(v.shape)}")

    # Expected shapes
    assert out['flow_coarse'].shape == (B, 2, *coarse_hw), f"coarse: {out['flow_coarse'].shape}"
    assert out['flow_mid'].shape == (B, 2, *mid_hw), f"mid: {out['flow_mid'].shape}"
    assert out['flow_fine'].shape == (B, 2, *fine_hw), f"fine: {out['flow_fine'].shape}"
    assert out['conf_fine'].shape == (B, 1, *fine_hw), f"conf: {out['conf_fine'].shape}"
    assert out['delta_xi'].shape == (B, 6), f"xi: {out['delta_xi'].shape}"
    assert 'flow_up' not in out, "flow_up should not exist (upsampler removed)"
    assert 'fine_flow_preds' in out, "fine_flow_preds missing (RAFT-style iterations)"
    assert len(out['fine_flow_preds']) == 4, f"expected 4 fine iterations, got {len(out['fine_flow_preds'])}"
    for i, fp in enumerate(out['fine_flow_preds']):
        assert fp.shape == (B, 2, *fine_hw), f"fine_flow_preds[{i}]: {fp.shape}"
    print(f"  fine_flow_preds: {len(out['fine_flow_preds'])} iterations, each {list(out['fine_flow_preds'][0].shape)}")

    print("\n✓ All shape checks passed!")

    # Gradient test
    print("\nBackward pass...")
    loss = out['flow_fine'].abs().mean() + out['delta_xi'].abs().mean()
    loss.backward()
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append((name, p.grad.norm().item()))
    print(f"  {len(grad_norms)} parameters have gradients")
    if grad_norms:
        max_name, max_norm = max(grad_norms, key=lambda x: x[1])
        print(f"  Max grad norm: {max_norm:.4f} ({max_name})")

    print("\n✓ Smoke test PASSED!")

if __name__ == '__main__':
    smoke_test()
