# 训练问题修复 - 快速指南

## 问题总结

修改后的训练无法收敛，原因是**6D旋转loss使用Frobenius范数（0-3）而非geodesic（0-180°），导致与平移loss量级失衡，Kendall's Loss无法正确学习权重**。

## 已修复的内容

### 1. rotation_6d loss改用geodesic
文件: `losses/pose_loss.py`
- 修改 `compute_rotation_loss_6d()` 使用geodesic distance
- 返回度数(0-180°)而非Frobenius范数(0-3)
- 保持与原始训练相同的loss量级

### 2. 配置恢复基础模式
文件: `configs/train_config.yaml`
```yaml
loss:
  use_kendall: false           # 先关闭，验证6D rotation基础功能
  use_relative_pose: false     # 先关闭，使用绝对位姿
  normalize_translation: false # 先关闭，保持原始量级
  rotation_loss: 'geodesic'    # geodesic返回度数
  rotation_weight: 1.0
  translation_weight: 1.0
```

### 3. 训练脚本支持开关
文件: `train.py`
- 添加 `use_kendall`、`use_relative_pose`、`normalize_translation` 的条件判断
- 根据配置选择正确的GT处理和loss计算方式
- 验证时正确处理归一化和相对位姿的反变换

## 立即开始训练

```bash
cd /home/yons/Projects/SplatLoc/implicit_correspondence
python train.py --config configs/train_config.yaml
```

## 预期效果

### Epoch 1
- 训练loss: rot=50-120度, trans=2-3米
- 验证角度误差: 约70°

### Epoch 5-10
- 角度误差应该降至20-30°

### Epoch 15-20
- 角度误差应该降至个位数（<10°）

**如果看到这个趋势 → 证明6D旋转本身work！**

## 如果成功，下一步

逐步启用高级特性（每次只改一个）：

### Step 1: 启用相对位姿
```yaml
use_relative_pose: true
```
预期：可能略微提升收敛速度

### Step 2: 启用归一化
```yaml
normalize_translation: true
```
预期：loss数值变化，但精度应保持

### Step 3: 启用Kendall's Loss
```yaml
use_kendall: true
init_log_var_rotation: 3.0
init_log_var_translation: -1.0
```
预期：自动平衡权重，可能进一步提升

## 如果仍然失败

### 检查1: 输出维度
```python
pose_9d.shape == (8, 9)  # batch_size=8
rotation_6d.shape == (8, 6)
```

### 检查2: 旋转矩阵有效性
```python
R = rotation_6d_to_matrix(rotation_6d)
det_R = torch.det(R)  # 应该≈1.0
```

### 检查3: Loss值
第一个batch应该看到：
```
rot_loss: 50-120 (度)
trans_loss: 2-3 (米)
```

如果rot_loss只有1-2 → loss计算有问题
如果trans_loss很大(>10) → 可能是数据问题

## 测试验证

```bash
python test_6d_rotation.py
```

应该看到：
```
✅ PASS: 6D旋转转换
✅ PASS: Kendall's Loss (rot_loss~135度)
✅ PASS: 相对位姿计算
✅ PASS: PoseRegressor输出
```

## 关键文件

- `losses/pose_loss.py` - PoseLossKendall.compute_rotation_loss_6d()
- `configs/train_config.yaml` - loss配置段
- `train.py` - train_epoch()和validate()
- `modules/pose_regressor.py` - rotation_6d_to_matrix()

## 监控指标

训练时观察：
1. `rot` loss: 应该50-120度范围（不是1-2！）
2. `trans` loss: 应该2-3米范围
3. 角度误差: 应该逐epoch下降
4. 梯度范数: 应该稳定（不爆炸，不消失）

## 文档

- `TRAINING_FAILURE_ANALYSIS.md` - 详细问题分析
- `ROTATION_6D_SUMMARY.md` - 6D旋转和改进总结
