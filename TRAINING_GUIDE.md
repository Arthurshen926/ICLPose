# 隐式对应关系位姿估计网络训练指南

## 📋 概述

本训练脚本用于训练隐式对应关系位姿估计网络（ICPoseNet），该网络基于SplatLoc的预训练模型进行位姿估计。

## 🚀 快速开始

### 1. 准备工作

确保你已经：
- ✅ 训练好SplatLoc模型（gaussians和feat_decoder）
- ✅ 准备好训练数据（RGB图像 + 相机位姿）
- ✅ 安装所有依赖包

### 2. 配置训练参数

编辑配置文件 `configs/train_config.yaml`：

```yaml
# 修改SplatLoc模型路径
splatloc:
  gaussians_path: "/path/to/point_cloud.ply"
  decoder_path: "/path/to/decoder.pth"

# 修改数据集路径
dataset:
  data_root: "/path/to/your/data"
  train_scene: "Sequence_1"
  val_scene: "Sequence_1"
```

### 3. 开始训练

**方式1: 使用启动脚本（推荐）**
```bash
cd /home/yons/Projects/SplatLoc/implicit_correspondence
chmod +x run_train.sh
./run_train.sh
```

**方式2: 直接使用Python**
```bash
cd /home/yons/Projects/SplatLoc/implicit_correspondence
python train.py --config configs/train_config.yaml
```

**方式3: 从checkpoint恢复训练**
```bash
python train.py \
    --config configs/train_config.yaml \
    --resume output/exp001/checkpoints/latest.pth
```

**方式4: 指定输出目录**
```bash
python train.py \
    --config configs/train_config.yaml \
    --output_dir output/exp002
```

## 📁 数据格式要求

训练数据应按以下结构组织：

```
data_root/
└── Sequence_1/
    ├── rgb/
    │   ├── rgb_0.png
    │   ├── rgb_1.png
    │   └── ...
    └── traj_w_c.txt  # 相机位姿文件
```

**位姿文件格式**：
- 每行16个数字，表示一个4x4变换矩阵（行优先）
- 每行对应一张图像的位姿

## ⚙️ 配置文件详解

### 模型配置
```yaml
model:
  feature_dim: 256      # 特征维度（需与decoder输出一致）
  num_queries: 128      # 可学习query数量
  fusion_layers: 6      # 跨模态融合层数
  num_heads: 8          # 注意力头数
  dropout: 0.1          # Dropout概率
```

### 训练配置
```yaml
training:
  num_epochs: 100           # 总epoch数
  batch_size: 8             # 批次大小
  learning_rate: 1.0e-4     # 学习率
  optimizer: 'adamw'        # 优化器类型
  scheduler: 'cosine'       # 学习率调度器
  grad_clip: 1.0            # 梯度裁剪
  val_interval: 1           # 验证间隔
  save_interval: 5          # checkpoint保存间隔
```

### 损失函数配置
```yaml
loss:
  rotation_loss: 'geodesic'     # 旋转损失: geodesic, l2, cosine, quaternion
  translation_loss: 'l2'        # 平移损失: l1, l2, smooth_l1
  rotation_weight: 1.0          # 旋转损失权重
  translation_weight: 1.0       # 平移损失权重
```

## 📊 训练输出

训练过程中会生成以下文件：

```
output/exp001/
├── checkpoints/
│   ├── latest.pth          # 最新checkpoint
│   ├── best.pth            # 最佳模型
│   ├── epoch_0005.pth      # 定期保存的checkpoint
│   ├── epoch_0010.pth
│   └── ...
└── logs/
    └── events.out.tfevents.xxx  # TensorBoard日志
```

### 查看训练日志

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir output/exp001/logs --port 6006
```

然后在浏览器打开：`http://localhost:6006`

## 📈 监控训练指标

训练过程中会显示以下指标：

**训练指标**：
- `loss`: 总损失
- `rotation_loss`: 旋转损失
- `translation_loss`: 平移损失
- `learning_rate`: 当前学习率

**验证指标**：
- `loss`: 总损失
- `rotation_loss`: 旋转损失
- `translation_loss`: 平移损失
- `rotation_error_mean`: 平均旋转角度误差（度）
- `rotation_error_median`: 中位数旋转角度误差（度）
- `translation_error_mean`: 平均平移距离误差（米）
- `translation_error_median`: 中位数平移距离误差（米）

## 🔧 常见问题

### 1. 内存不足

如果遇到CUDA内存不足，可以：
- 减小 `batch_size`
- 减小 `num_queries`
- 减小 `max_train_samples`

```yaml
training:
  batch_size: 4  # 从8改为4

model:
  num_queries: 64  # 从128改为64
```

### 2. 训练速度慢

可以尝试：
- 增加 `num_workers`
- 使用更小的验证集
- 减少验证频率

```yaml
training:
  num_workers: 8      # 增加数据加载线程
  val_interval: 5     # 每5个epoch验证一次

dataset:
  max_val_samples: 20  # 减少验证样本数
```

### 3. 损失不下降

可能的原因和解决方案：
- **学习率过大**：降低 `learning_rate`
- **权重未正确加载**：检查SplatLoc模型路径
- **数据问题**：检查数据格式和位姿文件

```yaml
training:
  learning_rate: 5.0e-5  # 降低学习率
```

### 4. 训练中断后恢复

使用 `--resume` 参数从checkpoint恢复：

```bash
python train.py \
    --config configs/train_config.yaml \
    --resume output/exp001/checkpoints/latest.pth
```

## 📝 训练技巧

### 1. 渐进式训练策略

**阶段1：冻结decoder，训练ICPoseNet**
```yaml
splatloc:
  freeze_decoder: true

training:
  num_epochs: 50
  learning_rate: 1.0e-4
```

**阶段2：微调decoder和ICPoseNet**
```yaml
splatloc:
  freeze_decoder: false

training:
  num_epochs: 50
  learning_rate: 1.0e-5
  decoder_lr: 1.0e-6
```

### 2. 损失函数选择

不同任务可以尝试不同的损失组合：

**方案1：测地距离（推荐用于大旋转）**
```yaml
loss:
  rotation_loss: 'geodesic'
  translation_loss: 'l2'
```

**方案2：四元数（推荐用于小旋转）**
```yaml
loss:
  rotation_loss: 'quaternion'
  translation_loss: 'l2'
  rotation_weight: 10.0  # 增大旋转权重
```

### 3. 学习率调度

**Cosine退火（推荐用于长时间训练）**
```yaml
training:
  scheduler: 'cosine'
  min_lr: 1.0e-6
```

**阶梯衰减（推荐用于快速收敛）**
```yaml
training:
  scheduler: 'step'
  lr_decay_step: 10
  lr_decay_gamma: 0.5
```

## 🎯 预期结果

在Replica数据集上，训练良好的模型应达到：
- 旋转误差：< 5度
- 平移误差：< 0.1米

## 📞 故障排查

如果遇到问题，请检查：

1. **检查数据路径**
   ```bash
   ls /path/to/data/Sequence_1/rgb/
   ls /path/to/data/Sequence_1/traj_w_c.txt
   ```

2. **检查SplatLoc模型**
   ```bash
   ls -lh /path/to/point_cloud.ply
   ls -lh /path/to/decoder.pth
   ```

3. **查看训练日志**
   ```bash
   tail -f output/exp001/logs/train.log
   ```

4. **验证配置文件**
   ```bash
   python -c "import yaml; print(yaml.safe_load(open('configs/train_config.yaml')))"
   ```

## 📚 相关文件

- `train.py`: 主训练脚本
- `configs/train_config.yaml`: 训练配置文件
- `run_train.sh`: 快速启动脚本
- `data/dataset.py`: 数据集类
- `losses/pose_loss.py`: 损失函数
- `models/ic_pose_net.py`: ICPoseNet模型

## 🔗 参考资料

- SplatLoc论文
- ICLPose论文
- PyTorch文档：https://pytorch.org/docs/

---

**最后更新**: 2026-01-17
