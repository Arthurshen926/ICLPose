# 可视化功能快速开始

## 30秒快速开始

### 1. 确认配置已启用可视化

检查 `train_config.yaml`:

```yaml
visualization:
  enable: true    # ← 确认是 true
  vis_interval: 5
  num_samples: 3
```

### 2. 运行训练

```bash
cd /home/yons/Projects/SplatLoc/implicit_correspondence
python train.py --config train_config.yaml
```

### 3. 查看可视化结果

```bash
# 可视化保存在这里
ls output/exp005/visualizations/
```

## 查看示例

测试可视化功能：

```bash
python test_visualization.py
# 查看生成的图像
ls -lh /tmp/test_*.png
```

## 可视化内容

每个epoch会保存3张图（每个样本）：

1. **sample_XX_pose.png** - 位姿预测 vs GT
2. **sample_XX_correspondence.png** - 2D-3D对应关系
3. **sample_XX_similarity.png** - 特征相似度矩阵

## 调整配置

### 更频繁保存
```yaml
vis_interval: 1  # 每个epoch保存
```

### 保存更多样本
```yaml
num_samples: 5  # 每次保存5个样本
```

### 禁用某些可视化
```yaml
vis_types:
  pose_prediction: true
  feature_similarity: false  # ← 禁用
  correspondence: true
```

## 下一步

- 阅读 [VISUALIZATION_README.md](./VISUALIZATION_README.md) 了解详细用法
- 查看 [VISUALIZATION_IMPLEMENTATION.md](./VISUALIZATION_IMPLEMENTATION.md) 了解实现细节

## 故障排除

**问题**: 没有生成可视化

**检查**:
1. `visualization.enable` 是否为 `true`
2. 是否到达 `vis_interval` 的epoch（如vis_interval=5，则epoch 5/10/15...才保存）
3. 检查 `visualizations/` 目录权限

**问题**: 可视化报错

**解决**: 运行测试脚本确认功能正常
```bash
python test_visualization.py
```

**问题**: 可视化图像不清晰

**调整**: 在 `utils/visualization.py` 中增加 `dpi`:
```python
plt.savefig(save_path, dpi=200, bbox_inches='tight')  # 从150改到200
```
