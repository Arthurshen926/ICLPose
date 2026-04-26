# 多GPU训练快速启动指南

## 一键启动多GPU训练

```bash
cd /home/yons/Projects/SplatLoc/implicit_correspondence
./train_distributed.sh
```

## 配置说明

### 当前配置（已优化）

- **GPU数量**：2张（CUDA:0 和 CUDA:1）
- **每GPU batch_size**：24（总batch_size = 48）
- **预计每GPU显存使用**：18-20GB（充分利用24GB显存）
- **数据加载线程**：8个（加速数据IO）

### 性能对比

| 配置 | 单GPU (之前) | 双GPU (现在) |
|------|-------------|-------------|
| batch_size (per GPU) | 8 | 24 |
| 总batch_size | 8 | 48 |
| 显存使用 | ~5GB | ~18-20GB |
| 显存利用率 | 20% | 75-83% |
| 预计训练速度 | 1× | ~3.6× |

速度提升来源：
- 2×GPU并行 ≈ 1.8-1.9× （考虑通信开销）
- 更大batch size（24 vs 8）≈ 2× （GPU利用率更高）
- 总体：约3.6-3.8倍加速

## 监控训练

### 查看GPU使用情况

```bash
watch -n 1 nvidia-smi
```

应该看到：
- 两张卡都在使用（GPU Util > 80%）
- 每张卡显存约18-20GB

### 查看训练日志

训练日志会实时打印，包括：
- 每个epoch的训练loss
- 每个epoch的验证loss和位姿误差
- 最佳模型保存信息

## 如果显存不足

如果遇到OOM（Out of Memory），修改配置：

```yaml
# configs/train_config.yaml
training:
  batch_size: 16  # 从24降到16
  val_batch_size: 8  # 从12降到8
```

## 恢复训练

如果训练中断，可以从checkpoint恢复：

```bash
# 修改train_distributed.sh，添加--resume参数
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --config configs/train_config.yaml \
    --resume output/exp003/checkpoints/latest.pth
```

## 验证配置是否正确

可以先用小数据集测试：

```yaml
# configs/train_config.yaml
dataset:
  max_train_samples: 100  # 只用100个样本测试
  max_val_samples: 20
```

训练几个epoch确认：
1. 两张卡都在工作
2. 显存使用正常（18-20GB）
3. loss正常下降
4. 没有报错

确认后再改回null使用全部数据。

## 预期训练时间

假设：
- 总样本数：900训练 + 50验证
- 总batch_size：48
- 每epoch迭代次数：900/48 ≈ 19 batches
- 每batch时间：约1-2秒
- 每epoch时间：约30-40秒

**100个epoch预计总时间：约50-70分钟**（对比之前单GPU约3-4小时）

## 与之前训练的兼容性

- ✅ 可以加载之前单GPU训练的checkpoint
- ✅ 训练结果应该一致（只是更快）
- ✅ 验证指标应该一致
- ✅ 可以从多GPU checkpoint恢复到单GPU继续训练

## 下一步优化（可选）

如果还想进一步加速：

### 1. 混合精度训练（FP16）
可以再提速30-50%，节省显存：
- 修改代码添加`torch.cuda.amp`支持
- 预计每GPU显存降到12-15GB
- 可以进一步增大batch_size到32-40

### 2. 增大batch_size
如果20GB还有余量，可以尝试：
- batch_size: 28 or 32 per GPU
- 预计显存：21-23GB

### 3. 调整学习率
更大的batch_size可能需要更大的学习率：
- 当前：5e-5（保守）
- 可以尝试：1e-4 或 2e-4
- 观察loss曲线，如果震荡就降回5e-5
