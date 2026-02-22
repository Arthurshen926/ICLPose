# 训练过程可视化功能说明

## 概述

为了更好地理解模型训练过程和隐式2D-3D对应关系学习，我们添加了可视化功能。在训练过程中，系统会定期保存以下可视化结果：

1. **位姿预测对比** - GT vs Pred位姿，旋转和平移误差
2. **2D-3D对应关系** - 2D特征点与3D点云的相似度可视化
3. **特征相似度矩阵** - 完整的2D-3D特征相似度热力图

## 配置

在 `train_config.yaml` 中配置可视化选项：

```yaml
visualization:
  enable: true              # 是否启用可视化
  vis_interval: 5           # 每隔多少个epoch保存可视化（建议5-10）
  num_samples: 3            # 每个epoch保存多少个样本
  vis_types:
    pose_prediction: true   # 位姿预测可视化
    feature_similarity: true  # 特征相似度矩阵
    correspondence: true    # 2D-3D对应关系
    query_evolution: false  # Query特征演化（需要sklearn，较慢）
```

## 输出结构

可视化结果保存在输出目录下的 `visualizations/` 文件夹：

```
output/exp005/
├── visualizations/
│   ├── epoch_0000/
│   │   ├── sample_00_pose.png
│   │   ├── sample_00_correspondence.png
│   │   ├── sample_00_similarity.png
│   │   ├── sample_01_pose.png
│   │   └── ...
│   ├── epoch_0005/
│   │   └── ...
│   └── epoch_0010/
│       └── ...
```

## 可视化类型详解

### 1. 位姿预测可视化 (pose_prediction)

**文件名**: `sample_XX_pose.png`

显示3个子图：
- **左图**: 输入图像
- **中图**: 旋转误差（度）
- **右图**: 平移误差（米）和GT/Pred的平移向量对比

**用途**: 直观看到模型预测的位姿与真实位姿的差异

### 2. 2D-3D对应关系可视化 (correspondence)

**文件名**: `sample_XX_correspondence.png`

显示3个子图：
- **左图**: 在图像上标注采样的2D特征点
- **中图**: 2D-3D平均相似度热力图/柱状图
- **右图**: 3D点云（按相似度着色）

**用途**: 理解2D图像特征与3D场景特征的隐式对应关系

### 3. 特征相似度矩阵 (feature_similarity)

**文件名**: `sample_XX_similarity.png`

显示2D特征(行) × 3D特征(列)的相似度矩阵热力图

**用途**: 
- 查看整体相似度分布
- 检查是否有明显的对应模式
- 诊断特征学习问题（如相似度过低/过高）

### 4. Query特征演化 (query_evolution) - 可选

**文件名**: `sample_XX_query_evolution.png`

通过PCA将query特征投影到2D空间，显示在不同融合层的分布变化

**用途**: 理解跨模态融合过程中query特征的演化

**注意**: 需要安装scikit-learn，且计算较慢，默认禁用

## 使用建议

### 训练监控

1. **初期训练** (epoch 0-20):
   - 重点看feature_similarity - 相似度应该逐渐出现模式
   - 检查correspondence - 2D-3D匹配是否合理

2. **中期训练** (epoch 20-100):
   - 关注pose_prediction - 误差应该持续下降
   - 观察对应关系是否更加清晰

3. **后期训练** (epoch 100+):
   - pose_prediction应该稳定在较低误差
   - correspondence应该显示明确的对应模式

### 问题诊断

| 观察现象 | 可能原因 | 建议 |
|---------|---------|------|
| 相似度矩阵接近随机 | 特征未学习 | 检查learning rate，增加训练时间 |
| 旋转误差不下降 | 旋转表示问题 | 检查旋转loss权重 |
| 平移误差过大 | 场景尺度问题 | 检查normalize_translation设置 |
| 2D-3D无明显对应 | 融合层数不足 | 增加fusion_layers |

## 示例分析

### 正常训练的可视化特征

✓ **相似度矩阵**: 出现块状或条状高亮区域  
✓ **对应关系**: 2D点能匹配到合理的3D区域  
✓ **位姿误差**: 旋转 < 10°, 平移 < 0.5m (视场景而定)  

### 异常训练的可视化特征

✗ **相似度矩阵**: 完全随机或全为高相似度  
✗ **对应关系**: 2D点匹配到无关的3D区域  
✗ **位姿误差**: 旋转 > 30°, 平移 > 2m (持续不下降)  

## 性能影响

- **内存**: 每个样本约1-2MB，对训练影响极小
- **速度**: 可视化在验证阶段进行，约增加5-10秒/epoch
- **建议**: 
  - `vis_interval=5` - 每5个epoch保存一次
  - `num_samples=3` - 每次保存3个样本
  - 禁用`query_evolution`（除非需要深入分析）

## 测试可视化

运行测试脚本验证可视化功能：

```bash
cd /home/yons/Projects/SplatLoc/implicit_correspondence
python test_visualization.py
```

这将在 `/tmp/` 目录生成测试可视化图像。

## 下一步计划

完成可视化集成后，建议的训练策略：

### 选项A: 使用全部数据（推荐）

```yaml
dataset:
  train_step: 1  # 使用全部900帧
  val_step: 1

training:
  num_epochs: 200
```

**预期性能**: 旋转误差 8-12°, 平移误差 0.8-1.0m

### 选项B: 保持采样 + 数据增强

```yaml
dataset:
  train_step: 5  # 180帧采样
  val_step: 1

training:
  num_epochs: 300
  # TODO: 添加数据增强配置
```

**预期性能**: 旋转误差 15-20°, 平移误差 ~1.0m

## 相关文件

- `utils/visualization.py` - 可视化函数实现
- `train.py` - 集成可视化到训练循环
- `train_config.yaml` - 可视化配置
- `test_visualization.py` - 可视化测试脚本
