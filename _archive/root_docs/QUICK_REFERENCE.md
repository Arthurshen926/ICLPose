# 训练快速参考

## 📝 文件说明

| 文件 | 说明 | 大小 |
|------|------|------|
| `train.py` | **主训练脚本** - 完整的训练流程 | 27KB |
| `configs/train_config.yaml` | **训练配置** - 所有训练参数 | 2.9KB |
| `run_train.sh` | **启动脚本** - 快速开始训练 | 737B |
| `check_environment.py` | **环境检查** - 训练前检查 | - |
| `TRAINING_GUIDE.md` | **详细指南** - 完整使用文档 | 6.9KB |
| `TRAINING_SUMMARY.md` | **实现总结** - 技术细节 | 8.6KB |
| `training_examples.sh` | **使用示例** - 各种启动方式 | - |

## 🚀 三步开始训练

```bash
# 1. 检查环境
python check_environment.py

# 2. 修改配置（编辑configs/train_config.yaml）
# 3. 开始训练
./run_train.sh
```

## 📋 常用命令

```bash
# 基本训练
python train.py --config configs/train_config.yaml

# 恢复训练
python train.py --config configs/train_config.yaml \
                --resume output/exp001/checkpoints/latest.pth

# 指定输出目录
python train.py --config configs/train_config.yaml \
                --output_dir output/exp002

# 查看TensorBoard
tensorboard --logdir output/exp001/logs --port 6006

# 后台运行
nohup python train.py --config configs/train_config.yaml > train.log 2>&1 &

# 在tmux中运行（推荐）
tmux new -s training
python train.py --config configs/train_config.yaml
# Ctrl+B, D 退出会话
# tmux attach -t training 重新进入
```

## ⚙️ 关键配置参数

### 必须修改的路径
```yaml
splatloc:
  gaussians_path: "/path/to/point_cloud.ply"
  decoder_path: "/path/to/decoder.pth"

dataset:
  data_root: "/path/to/your/data"
  train_scene: "Sequence_1"
```

### 常用调优参数
```yaml
training:
  batch_size: 8              # GPU内存不足时减小
  learning_rate: 1.0e-4      # 收敛慢时调大，震荡时调小
  num_epochs: 100            # 根据收敛情况调整

model:
  num_queries: 128           # 精度 vs 内存权衡
  fusion_layers: 6           # 精度 vs 速度权衡
```

## 📊 输出文件

```
output/exp001/
├── checkpoints/
│   ├── latest.pth          # 最新权重（每个epoch）
│   ├── best.pth            # 最佳权重（最低val loss）
│   └── epoch_XXXX.pth      # 定期保存（每N个epoch）
└── logs/
    └── events.out.tfevents.xxx  # TensorBoard日志
```

## 🎯 预期性能

**Replica数据集**：
- 旋转误差：< 5°
- 平移误差：< 0.1m
- 训练时间：2-3小时（RTX 3090, 100 epochs）
- 内存占用：8-12GB（batch_size=8）

## 🔍 监控指标

### 训练时查看
- `train/loss` - 应持续下降
- `train/learning_rate` - 按调度器变化

### 验证时查看
- `val/loss` - 应持续下降
- `val/rotation_error_mean` - 目标 < 5°
- `val/translation_error_mean` - 目标 < 0.1m

## ⚠️ 常见问题

### GPU内存不足
```yaml
training:
  batch_size: 4  # 减小batch size
model:
  num_queries: 64  # 减小query数量
```

### 训练不稳定
```yaml
training:
  learning_rate: 5.0e-5  # 降低学习率
  grad_clip: 1.0         # 启用梯度裁剪
```

### 收敛太慢
```yaml
training:
  learning_rate: 2.0e-4  # 提高学习率
  scheduler: 'step'      # 使用阶梯调度器
```

## 📚 文档索引

- **新手入门**: 阅读 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **技术细节**: 阅读 [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md)
- **代码示例**: 查看 [training_examples.sh](training_examples.sh)
- **环境检查**: 运行 `python check_environment.py`

## 🆘 获取帮助

```bash
# 查看命令行帮助
python train.py --help

# 验证配置文件
python -c "import yaml; print(yaml.safe_load(open('configs/train_config.yaml')))"

# 测试数据加载
python -c "from data.dataset import CorrespondenceDataset; \
           ds = CorrespondenceDataset('/path/to/data', 'Sequence_1'); \
           print(f'数据集大小: {len(ds)}')"
```

## 🎓 下一步

训练完成后：
1. 使用 `best.pth` 进行推理测试
2. 可视化预测的位姿
3. 在不同场景上评估泛化能力
4. 微调超参数优化性能

---

**版本**: v1.0  
**更新时间**: 2026-01-17  
**维护者**: ICPose Team
