# SLAM级别精度优化指南

## 当前状态分析（exp017 ~Epoch 60）

| 指标 | 当前值 | 目标值 | 差距 |
|------|--------|--------|------|
| 旋转误差 | 3.5° | <1° | 3.5x |
| 平移误差 | 0.5m | <0.1m | 5x |

## 为什么当前方法难以达到SLAM级别？

### 1. 特征分辨率限制
- 当前使用SplatLoc的HashGrid编码，分辨率可能不足
- 3DGS渲染的特征是"blended"的，精细几何信息丢失

### 2. 位姿回归 vs 几何求解
- 当前使用端到端回归，直接从特征预测位姿
- 缺少显式几何约束（如PnP求解器）

### 3. 训练数据限制
- 仅使用900帧Sequence_1训练
- 验证集Sequence_2可能有不同的位姿分布

## 推荐的改进策略

### 策略1: 添加可微分PnP层
```python
# 在最后阶段使用EPnP或PnP-RANSAC求解器
class DifferentiablePnP(nn.Module):
    def forward(self, img_keypoints, pcd_keypoints, intrinsics):
        # 使用soft correspondence权重
        # 调用kornia.geometry.epnp
        pass
```

**预期效果**: 旋转误差 <2°，平移误差 <0.2m

### 策略2: 增加训练数据
- 使用多个Sequence进行训练
- 添加更多数据增强（尤其是位姿扰动）

### 策略3: 调整位姿噪声范围
当前: rot=5°, trans=0.3m
如果目标是精细修正（<1°, <0.1m），应该减小训练噪声：
```yaml
pose_noise_rot_deg: 2.0    # 减小
pose_noise_trans_m: 0.1    # 减小
```

### 策略4: 课程学习
从小噪声开始训练，逐渐增大：
1. 阶段1 (0-100 epoch): rot=1°, trans=0.05m
2. 阶段2 (100-200 epoch): rot=2°, trans=0.1m  
3. 阶段3 (200-500 epoch): rot=5°, trans=0.3m

### 策略5: 多阶段Loss权重
在迭代模型中，对每个阶段单独计算loss，并加权：
```python
# 后期阶段权重更高（精细修正更重要）
stage_weights = [0.2, 0.3, 0.5]  # 3阶段
```

## 快速实验建议

### exp018: 减小训练噪声（测试精细修正能力）
```yaml
pose_noise_rot_deg: 2.0
pose_noise_trans_m: 0.1
```

### exp019: 添加课程学习
逐步增大噪声范围

### exp020: 集成可微分PnP
使用kornia的EPnP求解器作为最终精化步骤

## 长期方向

1. **特征增强**: 使用更高分辨率的图像特征（如DINO-v2）
2. **几何约束**: 集成深度监督或法向量约束
3. **测试时优化**: 在推理时进行微调（test-time adaptation）
