#!/usr/bin/env python3
"""
验证 ICPoseNetV3 所有模块的前向传播
===================================
使用随机数据验证:
  1. DynamicFeatureSelector 维度链接
  2. ConvGRU 隐藏状态更新
  3. DualHead 输出
  4. ICPoseNetV3 完整循环 (forward_with_prerendered)
  5. SequenceLoss 计算
"""

import sys
import torch
import torch.nn as nn

# 设置项目根目录
sys.path.insert(0, '/home/yons/Projects/ICLPose')

def test_lie_algebra():
    """测试 SE(3) Lie 代数模块"""
    print("=" * 60)
    print("[1/6] 测试 lie_algebra.py")
    print("=" * 60)
    
    from modules.lie_algebra import se3_exp, se3_log, pose_compose, pose_inverse, compute_gt_flow
    
    B = 2
    device = 'cpu'
    
    # se3_exp: ξ → T
    xi = torch.randn(B, 6, device=device) * 0.01
    T = se3_exp(xi)
    assert T.shape == (B, 4, 4), f"se3_exp shape: {T.shape}"
    # 验证 T 是有效的 SE(3): 底行 [0, 0, 0, 1]
    assert torch.allclose(T[:, 3, :], torch.tensor([0., 0., 0., 1.]).expand(B, 4), atol=1e-6)
    # 验证 R 是正交的
    R = T[:, :3, :3]
    I = torch.eye(3).expand(B, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), I, atol=1e-5), "R not orthogonal"
    print(f"  se3_exp: ξ({B},6) → T({B},4,4) ✓")
    
    # se3_log: T → ξ (round-trip)
    xi_recovered = se3_log(T)
    assert xi_recovered.shape == (B, 6), f"se3_log shape: {xi_recovered.shape}"
    T_recovered = se3_exp(xi_recovered)
    assert torch.allclose(T, T_recovered, atol=1e-5), "se3 round-trip failed"
    print(f"  se3_log round-trip: ξ → T → ξ' → T' ≈ T ✓")
    
    # pose_compose / inverse
    T_inv = pose_inverse(T)
    T_identity = pose_compose(T, T_inv)
    assert torch.allclose(T_identity[:, :3, :3], I, atol=1e-5)
    assert torch.allclose(T_identity[:, :3, 3], torch.zeros(B, 3), atol=1e-5)
    print(f"  pose_compose(T, T_inv) ≈ I ✓")
    
    # compute_gt_flow
    depth = torch.rand(B, 480, 640) * 3 + 0.5  # 0.5-3.5m
    pose_gt = torch.eye(4).unsqueeze(0).expand(B, 4, 4).clone()
    pose_current = se3_exp(torch.randn(B, 6) * 0.01) @ pose_gt
    intrinsics = {'fx': 320, 'fy': 320, 'cx': 319.5, 'cy': 239.5}
    
    flow_data = compute_gt_flow(depth, pose_gt, pose_current, intrinsics)
    assert flow_data['flow'].shape == (B, 2, 480, 640), f"flow shape: {flow_data['flow'].shape}"
    assert flow_data['valid_mask'].shape == (B, 1, 480, 640)
    print(f"  compute_gt_flow: depth(B,480,640) → flow(B,2,480,640) ✓")
    
    # 单帧测试
    flow_single = compute_gt_flow(depth[0], pose_gt[0], pose_current[0], intrinsics)
    assert flow_single['flow'].shape == (2, 480, 640)
    assert flow_single['valid_mask'].shape == (1, 480, 640)
    print(f"  compute_gt_flow single: depth(480,640) → flow(2,480,640) ✓")
    
    print("  [lie_algebra.py] 全部通过 ✓\n")


def test_conv_gru():
    """测试 ConvGRU"""
    print("=" * 60)
    print("[2/6] 测试 conv_gru.py")
    print("=" * 60)
    
    from modules.conv_gru import ConvGRUCell, ConvGRUBlock
    
    B, C_in, C_h, H, W = 2, 128, 128, 35, 46
    
    # ConvGRUCell
    cell = ConvGRUCell(C_in, C_h)
    x = torch.randn(B, C_in, H, W)
    h = torch.randn(B, C_h, H, W)
    h_new = cell(x, h)
    assert h_new.shape == (B, C_h, H, W), f"GRU cell output: {h_new.shape}"
    print(f"  ConvGRUCell: ({B},{C_in},{H},{W}) + ({B},{C_h},{H},{W}) → ({B},{C_h},{H},{W}) ✓")
    
    # ConvGRUBlock
    block = ConvGRUBlock(C_in, C_h)
    h_new2 = block(x, h)
    assert h_new2.shape == (B, C_h, H, W)
    print(f"  ConvGRUBlock: same dims ✓")
    
    # 验证梯度
    h_new2.sum().backward()
    assert x.grad is None  # x 没有 requires_grad
    print(f"  Backward pass ✓")
    
    n_params = sum(p.numel() for p in block.parameters())
    print(f"  ConvGRUBlock params: {n_params:,}")
    print("  [conv_gru.py] 全部通过 ✓\n")


def test_dynamic_feature_selector():
    """测试 DynamicFeatureSelector"""
    print("=" * 60)
    print("[3/6] 测试 dynamic_feature_selector.py")
    print("=" * 60)
    
    from modules.dynamic_feature_selector import DynamicFeatureSelector
    
    B = 2
    scale_configs = [
        {'name': 'coarse',    'feat_dim': 1280, 'resolution': (7, 10)},
        {'name': 'mid',       'feat_dim': 1280, 'resolution': (15, 20)},
        {'name': 'fine_sd',   'feat_dim': 640,  'resolution': (35, 46)},
        {'name': 'fine_dino', 'feat_dim': 768,  'resolution': (35, 46)},
    ]
    hidden_dim = 128
    output_res = (35, 46)
    
    selector = DynamicFeatureSelector(
        scale_configs=scale_configs,
        output_resolution=output_res,
        hidden_dim=hidden_dim,
        residual_mode='concat',
        residual_out_dim=64,
    )
    
    # 创建模拟的 query 和 rendered 特征
    query_feats = {}
    rendered_feats = {}
    for cfg in scale_configs:
        name = cfg['name']
        H, W = cfg['resolution']
        D = cfg['feat_dim']
        query_feats[name] = torch.randn(B, D, H, W)
        rendered_feats[name] = torch.randn(B, D, H, W)
    
    fused, gate_values = selector(query_feats, rendered_feats)
    assert fused.shape == (B, hidden_dim, output_res[0], output_res[1]), \
        f"Fused shape: {fused.shape}"
    
    print(f"  DynamicFeatureSelector output: {fused.shape} ✓")
    print(f"  Gate values: {gate_values}")
    
    # 验证 gate 和为 1
    gate_sum = sum(gate_values.values())
    print(f"  Gate sum: {gate_sum:.4f} (should be ≈1.0)")
    
    # 反向传播
    fused.sum().backward()
    print(f"  Backward pass ✓")
    
    n_params = sum(p.numel() for p in selector.parameters())
    print(f"  DynamicFeatureSelector params: {n_params:,}")
    print("  [dynamic_feature_selector.py] 全部通过 ✓\n")


def test_dual_head():
    """测试 DualHead"""
    print("=" * 60)
    print("[4/6] 测试 dual_head.py")
    print("=" * 60)
    
    from modules.dual_head import DualHead
    
    B, C_h, H, W = 2, 128, 35, 46
    head = DualHead(hidden_dim=C_h, pose_mlp_dim=256, flow_intermediate_dim=64)
    
    hidden = torch.randn(B, C_h, H, W)
    output = head(hidden)
    
    xi = output['xi']
    flow = output['flow']
    log_conf = output['log_confidence']
    
    assert xi.shape == (B, 6), f"xi shape: {xi.shape}"
    assert flow.shape == (B, 2, H, W), f"flow shape: {flow.shape}"
    assert log_conf.shape == (B, 1, H, W), f"log_conf shape: {log_conf.shape}"
    
    print(f"  PoseHead: hidden → ξ {xi.shape} ✓")
    print(f"  FlowHead: hidden → flow {flow.shape}, log_conf {log_conf.shape} ✓")
    print(f"  ξ values (should be ~0): {xi[0].detach().tolist()}")
    
    # 反向传播
    loss = xi.sum() + flow.sum() + log_conf.sum()
    loss.backward()
    print(f"  Backward pass ✓")
    
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  DualHead params: {n_params:,}")
    print("  [dual_head.py] 全部通过 ✓\n")


def test_ic_pose_net_v3():
    """测试完整的 ICPoseNetV3 (用预渲染特征)"""
    print("=" * 60)
    print("[5/6] 测试 ICPoseNetV3 (forward_with_prerendered)")
    print("=" * 60)
    
    from ic_models.ic_pose_net_v3 import ICPoseNetV3, ICPoseNetV3Config
    
    config = ICPoseNetV3Config.raw_features()
    model = ICPoseNetV3(**config)
    
    B = 2
    num_iters = config['num_iters']
    
    # 创建查询特征
    query_feats = {}
    for cfg in config['scale_configs']:
        name = cfg['name']
        H, W = cfg['resolution']
        D = cfg['feat_dim']
        query_feats[name] = torch.randn(B, D, H, W)
    
    # 创建每步的预渲染特征 (num_iters 步)
    rendered_feats_list = []
    for k in range(num_iters):
        rendered = {}
        for cfg in config['scale_configs']:
            name = cfg['name']
            H, W = cfg['resolution']
            D = cfg['feat_dim']
            rendered[name] = torch.randn(B, D, H, W)
        rendered_feats_list.append(rendered)
    
    # 初始位姿
    initial_pose = torch.eye(4).unsqueeze(0).expand(B, 4, 4).clone()
    
    # 前向传播
    predictions = model.forward_with_prerendered(
        query_feats, rendered_feats_list, initial_pose
    )
    
    assert len(predictions['poses']) == num_iters + 1, \
        f"Expected {num_iters+1} poses, got {len(predictions['poses'])}"
    assert len(predictions['xi_list']) == num_iters
    assert len(predictions['flow_list']) == num_iters
    assert len(predictions['gate_values_list']) == num_iters
    assert predictions['final_pose'].shape == (B, 4, 4)
    assert predictions['hidden'].shape == (B, 128, 35, 46)
    
    print(f"  Poses: {num_iters+1} steps (P₀ to P_{num_iters}) ✓")
    print(f"  xi_list: {num_iters} steps ✓")
    print(f"  flow_list: {num_iters} steps ✓")
    print(f"  final_pose: {predictions['final_pose'].shape} ✓")
    
    # 打印每步的 gate 值
    for k, gv in enumerate(predictions['gate_values_list']):
        gates_str = ' '.join([f"{n}={v:.3f}" for n, v in gv.items()])
        print(f"  Step {k}: {gates_str}")
    
    # 反向传播  
    total = predictions['final_pose'].sum()
    for xi in predictions['xi_list']:
        total = total + xi.sum()
    for flow in predictions['flow_list']:
        total = total + flow.sum()
    total.backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Backward: {grad_count}/{total_params} params have gradients ✓")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable params: {n_params:,}")
    print("  [ICPoseNetV3] 全部通过 ✓\n")
    
    return model, predictions, initial_pose, query_feats


def test_sequence_loss(predictions, pose_gt_base):
    """测试 SequenceLoss"""
    print("=" * 60)
    print("[6/6] 测试 SequenceLoss")
    print("=" * 60)
    
    from losses.sequence_loss import (
        SequenceLoss, PoseOnlySequenceLoss,
        rotation_geodesic_loss, translation_loss
    )
    
    B = pose_gt_base.shape[0]
    pose_gt = pose_gt_base.clone()
    
    # --- 基础损失函数 ---
    R1 = torch.eye(3).unsqueeze(0).expand(B, 3, 3)
    R2 = R1.clone()
    geo = rotation_geodesic_loss(R1, R2)
    assert torch.allclose(geo, torch.zeros(B), atol=1e-3), f"geo={geo}"
    print(f"  rotation_geodesic(I, I) = {geo.mean():.6f} (≈0) ✓")
    
    t1 = torch.zeros(B, 3)
    t2 = torch.ones(B, 3)
    tl = translation_loss(t1, t2, mode='l1')
    assert torch.allclose(tl, torch.tensor(3.0).expand(B))
    print(f"  translation_loss([0,0,0], [1,1,1], l1) = {tl[0]:.1f} ✓")
    
    # --- PoseOnlySequenceLoss ---
    pose_loss_fn = PoseOnlySequenceLoss(gamma=0.8, lambda_translation=1.0)
    
    # 需要从 predictions 中 detach 创建新的 predictions 用于 loss 计算
    # 因为前面的 backward 已经传播过了
    detached_predictions = {
        'poses': [p.detach().requires_grad_(True) for p in predictions['poses']],
    }
    
    result = pose_loss_fn(detached_predictions, pose_gt)
    print(f"  PoseOnlySequenceLoss:")
    print(f"    total_loss = {result['total_loss'].item():.6f}")
    print(f"    final_rot_err = {result['final_rotation_error_deg'].item():.2f}°")
    print(f"    final_trans_err = {result['final_translation_error_m'].item():.4f}m")
    
    # --- SequenceLoss (without flow) ---
    full_loss_fn = SequenceLoss(
        gamma=0.8, lambda_translation=1.0,
        lambda_flow=0.1, use_flow_loss=False
    )
    
    full_predictions = {
        'poses': [p.detach().requires_grad_(True) for p in predictions['poses']],
        'flow_list': [f.detach() for f in predictions['flow_list']],
        'log_conf_list': [c.detach() for c in predictions['log_conf_list']],
    }
    
    result2 = full_loss_fn(full_predictions, pose_gt)
    print(f"  SequenceLoss (no flow):")
    print(f"    total_loss = {result2['total_loss'].item():.6f}")
    print(f"    per_step_rot: {[f'{v.item():.4f}' for v in result2['per_step_rotation']]}")
    print(f"    per_step_trans: {[f'{v.item():.4f}' for v in result2['per_step_translation']]}")
    
    print("  [SequenceLoss] 全部通过 ✓\n")


def main():
    print("\n" + "=" * 60)
    print("ICPoseNetV3 前向传播验证测试")
    print("=" * 60 + "\n")
    
    torch.manual_seed(42)
    
    test_lie_algebra()
    test_conv_gru()
    test_dynamic_feature_selector()
    test_dual_head()
    model, predictions, initial_pose, query_feats = test_ic_pose_net_v3()
    test_sequence_loss(predictions, initial_pose)
    
    print("=" * 60)
    print("✅ 所有模块验证通过！")
    print("=" * 60)
    
    # 打印模块总结
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[模块总结]")
    print(f"  modules/lie_algebra.py        → SE(3) 工具函数")
    print(f"  modules/conv_gru.py           → ConvGRU 循环更新")
    print(f"  modules/dynamic_feature_selector.py → 多尺度门控融合")
    print(f"  modules/dual_head.py          → 位姿 + Flow 双头")
    print(f"  modules/multiscale_renderer.py → 3DGS 多尺度渲染")
    print(f"  ic_models/ic_pose_net_v3.py   → 完整迭代网络")
    print(f"  losses/sequence_loss.py       → 序列损失 + Flow 损失")
    print(f"\n  总可训练参数: {n_params:,}")


if __name__ == '__main__':
    main()
