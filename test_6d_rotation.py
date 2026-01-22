"""
测试6D旋转表示和相对位姿功能
"""

import torch
import numpy as np
from modules.pose_regressor import rotation_6d_to_matrix, matrix_to_rotation_6d
from losses.pose_loss import PoseLossKendall


def test_6d_rotation_conversion():
    """测试6D旋转表示的转换"""
    print("=" * 80)
    print("测试 1: 6D旋转表示转换")
    print("=" * 80)
    
    batch_size = 4
    
    # 生成随机旋转矩阵
    def random_rotation_matrix(B):
        # 使用随机四元数生成
        q = torch.randn(B, 4)
        q = q / q.norm(dim=1, keepdim=True)
        
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.zeros(B, 3, 3)
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        
        return R
    
    R_original = random_rotation_matrix(batch_size)
    print(f"原始旋转矩阵形状: {R_original.shape}")
    print(f"第一个矩阵的行列式: {torch.det(R_original[0]):.6f} (应接近1.0)")
    
    # 转换为6D表示
    d6 = matrix_to_rotation_6d(R_original)
    print(f"\n6D表示形状: {d6.shape}")
    print(f"6D表示范围: [{d6.min():.3f}, {d6.max():.3f}]")
    
    # 转换回旋转矩阵
    R_reconstructed = rotation_6d_to_matrix(d6)
    print(f"\n重建旋转矩阵形状: {R_reconstructed.shape}")
    print(f"重建矩阵的行列式: {torch.det(R_reconstructed[0]):.6f}")
    
    # 计算重建误差
    error = (R_original - R_reconstructed).abs().max().item()
    print(f"\n最大重建误差: {error:.8f}")
    
    if error < 1e-5:
        print("✅ 6D旋转转换测试通过！")
    else:
        print(f"❌ 6D旋转转换测试失败！误差过大: {error}")
    
    return error < 1e-5


def test_kendall_loss():
    """测试Kendall's Loss"""
    print("\n" + "=" * 80)
    print("测试 2: Kendall's Loss")
    print("=" * 80)
    
    batch_size = 8
    
    # 创建loss函数
    loss_fn = PoseLossKendall(
        rotation_loss_type='rotation_6d',
        translation_loss_type='l2',
        reduction='mean',
        init_log_var_rotation=0.0,
        init_log_var_translation=0.0,
    )
    
    print(f"\n初始log_var_rotation: {loss_fn.log_var_rotation.item():.3f}")
    print(f"初始log_var_translation: {loss_fn.log_var_translation.item():.3f}")
    print(f"初始权重_rotation: {torch.exp(-loss_fn.log_var_rotation).item():.3f}")
    print(f"初始权重_translation: {torch.exp(-loss_fn.log_var_translation).item():.3f}")
    
    # 创建假数据
    def random_rotation_matrix(B):
        q = torch.randn(B, 4)
        q = q / q.norm(dim=1, keepdim=True)
        
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.zeros(B, 3, 3)
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        
        return R
    
    R_gt = random_rotation_matrix(batch_size)
    t_gt = torch.randn(batch_size, 3) * 0.5
    
    R_pred = random_rotation_matrix(batch_size)
    t_pred = torch.randn(batch_size, 3) * 0.5
    
    # 构建位姿矩阵
    pose_gt = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    pose_gt[:, :3, :3] = R_gt
    pose_gt[:, :3, 3] = t_gt
    
    # 前向传播
    losses = loss_fn(
        pose_pred=(R_pred, t_pred),
        pose_gt=pose_gt,
        return_components=True
    )
    
    print(f"\n损失计算结果:")
    print(f"  总损失: {losses['loss'].item():.4f}")
    print(f"  旋转损失（原始）: {losses['rotation_loss'].item():.4f}")
    print(f"  平移损失（原始）: {losses['translation_loss'].item():.4f}")
    print(f"  旋转损失（加权）: {losses['weighted_rotation_loss'].item():.4f}")
    print(f"  平移损失（加权）: {losses['weighted_translation_loss'].item():.4f}")
    
    # 测试梯度
    loss = losses['loss']
    loss.backward()
    
    print(f"\nlog_var梯度:")
    print(f"  log_var_rotation梯度: {loss_fn.log_var_rotation.grad.item():.6f}")
    print(f"  log_var_translation梯度: {loss_fn.log_var_translation.grad.item():.6f}")
    
    if loss.item() > 0 and not torch.isnan(loss) and not torch.isinf(loss):
        print("✅ Kendall's Loss测试通过！")
        return True
    else:
        print("❌ Kendall's Loss测试失败！")
        return False


def test_relative_pose():
    """测试相对位姿计算"""
    print("\n" + "=" * 80)
    print("测试 3: 相对位姿计算")
    print("=" * 80)
    
    from train import compute_relative_pose, compose_pose
    
    batch_size = 4
    
    # 创建初始位姿（单位矩阵 + 小位移）
    pose_init = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    pose_init[:, :3, 3] = torch.tensor([0.0, 0.0, 0.0])
    
    # 创建目标位姿（旋转 + 平移）
    pose_target = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    # 绕Z轴旋转45度
    angle = torch.tensor([0.0, 0.1, 0.2, 0.3])  # 不同的旋转角度
    for i in range(batch_size):
        c = torch.cos(angle[i])
        s = torch.sin(angle[i])
        pose_target[i, 0, 0] = c
        pose_target[i, 0, 1] = -s
        pose_target[i, 1, 0] = s
        pose_target[i, 1, 1] = c
    pose_target[:, :3, 3] = torch.randn(batch_size, 3) * 0.5
    
    print(f"初始位姿平移: {pose_init[0, :3, 3]}")
    print(f"目标位姿平移: {pose_target[0, :3, 3]}")
    
    # 计算相对位姿
    pose_rel = compute_relative_pose(pose_target, pose_init)
    print(f"\n相对位姿平移: {pose_rel[0, :3, 3]}")
    
    # 重建目标位姿
    pose_reconstructed = compose_pose(pose_rel, pose_init)
    
    # 计算重建误差
    error = (pose_target - pose_reconstructed).abs().max().item()
    print(f"\n最大重建误差: {error:.8f}")
    
    if error < 1e-5:
        print("✅ 相对位姿计算测试通过！")
        return True
    else:
        print(f"❌ 相对位姿计算测试失败！误差过大: {error}")
        return False


def test_pose_regressor_output():
    """测试PoseRegressor输出维度"""
    print("\n" + "=" * 80)
    print("测试 4: PoseRegressor输出维度")
    print("=" * 80)
    
    from modules.pose_regressor import PoseRegressor
    
    batch_size = 4
    num_queries = 128
    feature_dim = 256
    
    # 创建模型
    model = PoseRegressor(
        feature_dim=feature_dim,
        hidden_dim=512,
        output_dim=9  # 3平移 + 6旋转
    )
    
    # 创建假输入
    fused_feats = torch.randn(batch_size, num_queries, feature_dim)
    
    # 前向传播
    pose, rotation_6d, translation = model(fused_feats)
    
    print(f"输入特征形状: {fused_feats.shape}")
    print(f"输出位姿形状: {pose.shape} (应为 {batch_size}, 9)")
    print(f"输出旋转6D形状: {rotation_6d.shape} (应为 {batch_size}, 6)")
    print(f"输出平移形状: {translation.shape} (应为 {batch_size}, 3)")
    
    # 验证维度
    success = (
        pose.shape == (batch_size, 9) and
        rotation_6d.shape == (batch_size, 6) and
        translation.shape == (batch_size, 3)
    )
    
    if success:
        print("✅ PoseRegressor输出维度测试通过！")
    else:
        print("❌ PoseRegressor输出维度测试失败！")
    
    # 测试6D转旋转矩阵
    R = rotation_6d_to_matrix(rotation_6d)
    print(f"\n旋转矩阵形状: {R.shape} (应为 {batch_size}, 3, 3)")
    print(f"旋转矩阵行列式: {torch.det(R[0]):.6f} (应接近1.0)")
    
    det_error = abs(torch.det(R[0]).item() - 1.0)
    if det_error < 0.1:
        print("✅ 6D->旋转矩阵转换正常！")
    else:
        print(f"⚠️ 6D->旋转矩阵转换可能有问题，行列式偏差: {det_error}")
    
    return success


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "6D旋转和相对位姿测试套件" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    
    results = []
    
    # 测试1: 6D旋转转换
    results.append(("6D旋转转换", test_6d_rotation_conversion()))
    
    # 测试2: Kendall's Loss
    results.append(("Kendall's Loss", test_kendall_loss()))
    
    # 测试3: 相对位姿计算
    results.append(("相对位姿计算", test_relative_pose()))
    
    # 测试4: PoseRegressor输出
    results.append(("PoseRegressor输出", test_pose_regressor_output()))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 所有测试通过！可以开始训练。")
    else:
        print("⚠️ 部分测试失败，请检查实现。")
    print("=" * 80)
    
    return all_passed


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
