# 多GPU分布式训练配置说明

## 概览

已添加多GPU分布式训练支持，可以充分利用2张24GB显卡的显存。

## 主要修改

### 1. train.py修改

- **分布式训练支持**：添加了`torch.distributed`、`DistributedDataParallel`和`DistributedSampler`
- **ICPoseTrainer类**：
  - 添加`local_rank`参数（-1表示单GPU，>=0表示分布式训练）
  - 添加`is_distributed`和`is_main_process`标志
  - 使用DDP包装模型
  - 只在主进程（rank 0）进行日志记录、checkpoint保存和打印
  
- **DataLoader修改**：
  - 使用`DistributedSampler`分配数据到不同GPU
  - 训练时shuffle=False（sampler处理shuffle）
  - 验证时也使用DistributedSampler保证数据完整覆盖
  
- **训练循环修改**：
  - `train_epoch`：每个epoch开始时调用`sampler.set_epoch()`保证不同epoch的shuffle不同
  - `validate`：使用`dist.all_reduce`同步所有GPU的统计数据，在主进程汇总结果
  - checkpoint保存：只在主进程保存
  - 日志打印：只在主进程打印

### 2. train_config.yaml修改

```yaml
training:
  batch_size: 24        # 每GPU的batch size（之前是8）
  val_batch_size: 12    # 每GPU的验证batch size（之前是4）
  num_workers: 8        # 数据加载线程数（之前是4）
```

**总batch size计算**：
- 训练：24 (per GPU) × 2 (GPUs) = **48**
- 验证：12 (per GPU) × 2 (GPUs) = **24**

### 3. 启动脚本

创建了`train_distributed.sh`用于启动分布式训练：

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --config configs/train_config.yaml
```

## 使用方法

### 多GPU训练（推荐）

```bash
cd /home/yons/Projects/SplatLoc/implicit_correspondence
./train_distributed.sh
```

或者直接使用torchrun：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --config configs/train_config.yaml
```

### 单GPU训练（向后兼容）

原来的单GPU训练方式仍然可用：

```bash
python train.py --config configs/train_config.yaml
```

## 显存使用估算

### 之前（单GPU，batch_size=8）
- 显存使用：~5GB
- 利用率：5GB / 24GB ≈ 20%

### 现在（双GPU，batch_size=24 per GPU）
- 预计每GPU显存：~18-20GB
- 利用率：18-20GB / 24GB ≈ 75-83%

### 显存占用分析
- **模型参数**：ICPoseNet 9.2M参数 ≈ 37MB（FP32）
- **特征解码器**：FeatureDecoder（冻结）≈ 50MB
- **Gaussian点云**：约500MB
- **每个样本**：
  - 图像特征：128 × 256 ≈ 0.13MB
  - 点云特征：128 × 256 ≈ 0.13MB
  - 中间激活：约1-2GB（取决于batch size）
- **梯度和优化器状态**：约2×模型大小 ≈ 150MB

batch_size=24时，预计总显存 ≈ 18-20GB/GPU

## 性能提升预期

### 训练速度
- **并行加速**：2× GPU → 约1.8-1.9×加速（考虑通信开销）
- **更大batch size**：24 vs 8 → 更稳定的梯度估计
- **更快数据加载**：8个worker vs 4个 → 减少IO瓶颈

### 训练质量
- **更大batch size**：
  - 更稳定的梯度估计
  - 可能需要调整学习率（当前保持5e-5）
  - BN层统计更准确（如果使用）

## 注意事项

### 1. batch size与学习率

当前配置：
- batch_size: 24 per GPU（总48）
- learning_rate: 5e-5

根据"线性缩放规则"，batch size从8增大到48（6倍），理论上可以按比例增大学习率到3e-4。但考虑到：
- 当前学习率已经调低以提高稳定性
- 建议先用5e-5训练几个epoch观察效果
- 如果收敛太慢，可以尝试增大到1e-4

### 2. 梯度累积（可选优化）

如果24GB显存不够，可以减小batch_size并使用梯度累积：

```python
# 在train_epoch中添加
accumulation_steps = 2  # 每2步更新一次
for batch_idx, batch in enumerate(pbar):
    loss = ... / accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        self.optimizer.step()
        self.optimizer.zero_grad()
```

### 3. 监控显存使用

训练开始后可以用nvidia-smi监控：

```bash
watch -n 1 nvidia-smi
```

### 4. 调试建议

第一次运行时建议：
1. 先用小数据集测试（如max_train_samples: 100）
2. 观察显存使用是否正常
3. 确认两张卡都在工作（nvidia-smi查看utilization）
4. 检查loss是否正常下降

## 故障排查

### 问题1：CUDA out of memory
**解决**：减小batch_size（改为16或20）

### 问题2：进程hang住不动
**原因**：可能是分布式通信问题
**解决**：
- 检查防火墙设置
- 尝试更换master_port（如29501）
- 确认NCCL安装正确

### 问题3：只有一个GPU在工作
**原因**：可能是CUDA_VISIBLE_DEVICES设置问题
**解决**：
```bash
echo $CUDA_VISIBLE_DEVICES  # 应该输出 0,1
nvidia-smi  # 确认两张卡都可见
```

### 问题4：精度loss与单GPU不一致
**原因**：DistributedSampler的shuffle和drop_last设置
**解决**：已正确配置，两者应该一致

## 与原训练的兼容性

- ✅ checkpoint格式不变
- ✅ 配置文件向后兼容
- ✅ 可以从单GPU checkpoint恢复到多GPU训练
- ✅ 可以从多GPU checkpoint恢复到单GPU训练
- ✅ 验证结果应该一致（考虑舍入误差）

## 后续优化建议

### 1. 混合精度训练（AMP）
使用torch.cuda.amp可以进一步提高速度和节省显存：

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 在训练循环中
with autocast():
    loss = ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 梯度检查点（Gradient Checkpointing）
如果模型太大，可以用梯度检查点节省显存：

```python
from torch.utils.checkpoint import checkpoint
# 在模型forward中使用checkpoint包装某些层
```

### 3. 更大的batch size测试
可以尝试：
- batch_size=32 per GPU（总64）
- batch_size=40 per GPU（总80）

观察显存使用和训练效果。
