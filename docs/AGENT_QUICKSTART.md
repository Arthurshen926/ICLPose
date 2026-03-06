# Agent 快速上手指南

> 给接手的 AI Agent 的操作手册 — 最短路径理解和修改项目

---

## 必看文件 (按优先级)

1. **`scripts/train_ms_flow.py`** — 训练入口, 包含完整训练逻辑
2. **`ic_models/ms_flow_pose_net.py`** — 核心模型 (~676行, 必读)
3. **`configs/exp032_cosine_fiters8.yaml`** — 当前最佳配置
4. **`modules/geometry_solver.py`** — Image Jacobian 几何求解 (~150行)
5. **`data/dataset_v4.py`** — 数据加载 (~273行)
6. **`modules/multiscale_renderer.py`** — 多尺度渲染 (~336行)
7. **`modules/lie_algebra.py`** — SE(3) 操作 (~342行)

## 一分钟理解项目

```
查询图像特征 + 初始位姿 → [重复3-5次] → 精确位姿
                              ↓
                    用当前位姿渲染参考特征
                              ↓
                    多尺度光流匹配 (粗→细)
                              ↓
                    几何求解 → 位姿增量
                              ↓
                    更新当前位姿, 回到顶部
```

## 启动训练

```bash
# 激活环境
conda activate geo-aware

# 单GPU训练 (当前标准做法)
CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py \
    --config configs/exp032_cosine_fiters8.yaml

# 监控训练
grep '\[Val E' output/exp032_train.log | tail -10
grep '★' output/exp032_train.log  # 看最佳结果
```

## 创建新实验

```bash
# 1. 复制配置
cp configs/exp032_cosine_fiters8.yaml configs/exp033_your_change.yaml

# 2. 修改关键字段
# experiment_name: exp033_your_change
# warmstart: output/exp032/best.pth  (从上一个最佳启动)
# 调整你想改的超参

# 3. 启动
CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py \
    --config configs/exp033_your_change.yaml
```

## 常见修改场景

### 改模型结构
文件: `ic_models/ms_flow_pose_net.py`
- `ScaleDecoder`: 特征降维 (改通道数、层数)
- `FineDualDecoder`: Fine 尺度 SD+DINO 融合
- `FlowRefinementHead`: RAFT GRU 精化 (改 hidden_dim、迭代次数)
- `MSFlowPoseNet.forward()`: 整体前向逻辑

### 改损失函数
文件: `scripts/train_ms_flow.py`
- `multiscale_flow_loss()`: 光流监督
- `pose_loss()`: 位姿监督 (cosine/acos rotation + L2 translation)

### 改训练策略
文件: `scripts/train_ms_flow.py` 中 `MSFlowTrainer` 类
- `_train_step()`: 外循环迭代逻辑
- `_update_noise_for_epoch()`: 噪声课程
- `_effective_pose_weight()`: Pose loss 暖启动

### 改渲染
文件: `modules/multiscale_renderer.py`
- `render_batch()`: 多尺度渲染
- 改分辨率: 修改 3DGS 模型的渲染参数

## 当前瓶颈与机会

| 优化方向 | 难度 | 预期收益 |
|---------|------|---------|
| 增加 fine_iters (12→16) | 低 | 可能进一步降低 rot |
| Multi-GPU DDP | 中 | 训练速度 ×N |
| 更大噪声鲁棒性 | 中 | 泛化能力 |
| 新场景 (room_1, room_2) | 低 | 验证泛化 |
| NetVLAD 初始位姿 | 中 | 更真实的验证 |
| Online 特征提取 | 高 | 减少离线预处理 |

## 重要提醒

1. **CUDA 版本**: 4090 需要 CUDA 11.8+, 当前是 11.6 (见 `docs/ENVIRONMENT_SETUP.md`)
2. **single GPU**: 当前训练脚本不支持多 GPU, batch_size=1 占 ~22GB
3. **Warmstart**: 新实验务必从 `output/exp032/best.pth` 启动
4. **FP32 pose loss**: 位姿损失必须在 fp32 下计算, 否则 NaN
5. **外循环 detach**: `T_curr = (delta_T @ T_curr).detach()` — 不要删除 detach!

---

## ⚠️ 另一条工作线: OldHospital 2DGS 几何重建

> 本项目有两条并行工作线。上面内容覆盖的是**定位网络**。

另一条同等重要的工作线是 **OldHospital 2DGS 几何重建 + WildGaussians 外观建模**:

- **目标**: ≥20 dB PSNR | **当前最佳**: 18.21 dB (retrain38f)
- **核心脚本**: `feature_3dgs/train_2dgs_geometry.py` (~2700 行, 独立完整)
- **数据**: `dataset/OldHospital/` (~32 GB, 895 train / 182 test)

### 必看文件 (2DGS)

1. **`docs/OLDHOSPITAL_2DGS_GUIDE.md`** — ★ 完整技术文档 (架构/实验历史/训练配置/优化方向)
2. **`feature_3dgs/train_2dgs_geometry.py`** — 唯一核心文件 (模型+训练+评估全在里面)
3. **`scripts/auto_retrain38g.sh`** — 自动训练脚本 (含最新配置参考)

### 快速启动 2DGS

```bash
# 评估现有最佳 checkpoint
CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain38f \
    --use_mask --wildgaussians --eval_only --checkpoint_iter 30000 --wg_test_opt_steps 100

# 新训练 (参见 docs/OLDHOSPITAL_2DGS_GUIDE.md §7 完整参数)
```
