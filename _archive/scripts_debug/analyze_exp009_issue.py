"""
分析EXP009验证平移误差问题

观察到的现象：
- 训练loss稳定下降
- 验证loss也在下降
- 但是验证平移误差先变小后变大（从~6.5m降到最低后又升到6.8m+）

可能的原因分析
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt

# 1. 检查配置
print("=" * 80)
print("EXP009配置分析")
print("=" * 80)

config_path = "configs/train_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("\n关键配置:")
print(f"  use_relative_pose: {config['loss']['use_relative_pose']}")
print(f"  normalize_translation: {config['loss']['normalize_translation']}")
print(f"  use_kendall: {config['loss']['use_kendall']}")
print(f"  init_log_var_rotation: {config['loss']['init_log_var_rotation']}")
print(f"  init_log_var_translation: {config['loss']['init_log_var_translation']}")

print("\n" + "=" * 80)
print("问题分析")
print("=" * 80)

print("""
观察到的现象（从tensorboard图表）：
1. 训练Loss: 稳定下降（从800+降到20左右）✓
2. 验证Loss: 也在下降（从90+降到30左右）✓
3. 训练平移误差: 稳定下降（降到~0.3m）✓
4. 验证平移误差: 先降后升（6.5m → 最低点 → 6.8m+）❌

可能的原因：

【原因1】过拟合 - 模型记忆训练数据
- 症状: 训练误差很低（0.3m），验证误差很高（6.8m）
- 原因: 缺少正则化、模型容量过大
- EXP009问题: 没有Self-Attention，模型表达能力弱，容易记忆

【原因2】Kendall Loss权重不平衡导致的偏向优化
- log_var_rotation = 0.0 → weight_rot = exp(0) = 1.0
- log_var_translation = 0.0 → weight_trans = exp(0) = 1.0
- 但Kendall是可学习的！随着训练，权重会变化
- 如果模型学到"优化旋转更容易"，可能会降低translation权重

【原因3】normalize_translation的副作用
- 当前配置: normalize_translation = true
- 效果: translation除以中位数尺度（约3-5m）
- 训练时loss看的是归一化后的值
- 但验证误差看的是原始尺度的绝对误差
- 如果归一化破坏了平移的真实分布，可能导致误差计算偏差

【原因4】没有初始位姿导致的绝对位姿学习困难
- 当前配置: use_relative_pose = false（学习绝对位姿）
- 问题: 绝对位姿范围很大，难以学习
- 房间坐标系: x ∈ [-1, 7], y ∈ [-1.3, 3.7], z ∈ [-1.7, 1.4]
- 平移需要学习到米级别的绝对坐标，而不是相对增量

【原因5】验证集和训练集分布不一致
- 训练集: Sequence_1
- 验证集: Sequence_2
- 如果两个序列的相机轨迹、场景覆盖范围差异大，模型泛化困难

【原因6】Kendall Loss的log_var发散
- Kendall Loss中的log_var是可学习参数
- 如果log_var_translation持续增大 → 平移权重exp(-log_var)降低
- 模型会"放弃"优化平移，专注于旋转

分析优先级：
1. ⭐⭐⭐ 检查Kendall Loss的权重变化（log_var_rotation, log_var_translation）
2. ⭐⭐⭐ 改用相对位姿（use_relative_pose=true）
3. ⭐⭐ 检查训练/验证集的平移分布
4. ⭐ 关闭normalize_translation，使用原始尺度
5. ⭐ 添加正则化（weight decay, dropout）
""")

print("\n" + "=" * 80)
print("建议修复方案")
print("=" * 80)

print("""
【方案1】启用相对位姿（推荐，最可能解决问题）
修改配置:
  use_relative_pose: true  # 改为true

原理:
- 不学习绝对位姿（难），而是学习相对变换（易）
- 相对位姿范围小，更容易收敛
- SLAM/tracking场景的标准做法

训练/验证时的处理:
- 训练: pose_rel = pose_target @ inv(pose_init)
- 验证: pose_abs = pose_init @ pose_rel (恢复绝对位姿计算误差)

【方案2】关闭normalize_translation
修改配置:
  normalize_translation: false  # 改为false

原理:
- 避免归一化引入的尺度偏差
- Kendall Loss已经可以自动平衡权重，不需要手动归一化

【方案3】固定Kendall Loss权重（不让它学习）
修改代码:
- 在KendallLoss中，设置log_var.requires_grad = False
- 或者直接改回加权Loss

【方案4】EXP010的Self-Cross架构（已完成）
- 新架构解决了queries无法交互的问题
- 期待泛化能力提升

【方案5】监控Kendall权重
添加到training.log:
- 每个epoch记录: weight_rotation, weight_translation
- 观察是否有一个权重持续下降

建议执行顺序:
1. 先启动EXP010（新架构 + 现有配置）→ 观察是否改善
2. 如果EXP010还有问题，尝试 use_relative_pose=true
3. 添加Kendall权重监控，确认是否有权重失衡
""")

print("\n" + "=" * 80)
print("关于初始位姿的回答")
print("=" * 80)

print("""
Q: 训练和验证阶段是否都有初始位姿？

A: 当前配置下（use_relative_pose=false）:

【训练阶段】:
- ❌ 没有使用初始位姿
- 模型直接预测绝对位姿: T_pred ∈ SE(3)
- GT也是绝对位姿: T_gt ∈ SE(3)
- Loss = |T_pred - T_gt|

【验证阶段】:
- ❌ 没有使用初始位姿
- 评估方式与训练一致
- 计算绝对位姿误差

如果改为 use_relative_pose=true:

【训练阶段】:
- ✅ 使用batch第一帧作为初始位姿
- pose_init = batch[0].pose
- 模型预测相对变换: T_rel_pred
- GT计算相对变换: T_rel_gt = T_target @ inv(T_init)
- Loss = |T_rel_pred - T_rel_gt|

【验证阶段】:
- ✅ 同样使用batch第一帧作为初始位姿
- 评估时转回绝对位姿: T_pred_abs = T_init @ T_rel_pred
- 计算绝对位姿误差

代码位置:
- train.py line 785: pose_init = gt_poses_abs[0:1].expand(...)
- train.py line 787: gt_poses = compute_relative_pose(gt_poses_abs, pose_init)
- train.py line 1018: pose_pred_eval = compose_pose(pose_pred_eval, pose_init)

结论:
- 当前EXP009: 没有初始位姿，学习绝对位姿（困难）
- 建议改为: 使用初始位姿，学习相对变换（容易）
""")
