# ICLPose 项目迁移指南 (6×4090 服务器)

> **最后更新**: 2025-01 | **分支**: `v2-iterative-routing` | **最新实验**: exp032 (rot=0.33°, <1°=95.6%)

---

## 目录

1. [项目概述](#1-项目概述)
2. [代码结构总览](#2-代码结构总览)
3. [环境配置 (4090 适配)](#3-环境配置)
4. [核心架构详解](#4-核心架构详解)
5. [训练流程](#5-训练流程)
6. [实验历史与改进](#6-实验历史与改进)
7. [当前最佳结果](#7-当前最佳结果)
8. [后续优化方向](#8-后续优化方向)
9. [快速上手](#9-快速上手)

---

## 1. 项目概述

### 1.1 目标

ICLPose 是一个基于 **3DGS (3D Gaussian Splatting) 特征图** 的 **6-DOF 相机位姿估计** 系统。核心思想:

1. **离线阶段**: 用预训练的 Stable Diffusion / DINOv2 提取多尺度特征，训练 3DGS 特征场 (Feature 3DGS)
2. **在线阶段**: 给定查询图像的特征 + 初始位姿估计，通过 **可微渲染 + 光流 + 几何求解** 迭代优化位姿

### 1.2 当前主力模型

**MSFlowPoseNet** — Multi-Scale Flow Pose Network:
- 多尺度 coarse-to-fine 光流匹配 (7×10 → 15×20 → 35×46)
- 外循环迭代精化 (Outer-loop iterative refinement): 每次用当前位姿重新渲染特征图
- RAFT-style GRU 细粒度精化
- Image Jacobian + 加权最小二乘几何求解 → 6-DOF 位姿增量

### 1.3 数据集

当前使用 **Replica room_0**:
- Sequence_1: 训练集 (2000 帧)
- Sequence_2: 验证集 (900 帧)
- 相机内参: fx=fy=320.0, cx=319.5, cy=239.5, 640×480
- 特征分辨率: coarse 7×10, mid 15×20, fine 35×46

---

## 2. 代码结构总览

```
ICLPose/
├── scripts/
│   └── train_ms_flow.py          ★ 主训练脚本 (~887行, MSFlowPoseNet专用)
├── configs/
│   ├── exp032_cosine_fiters8.yaml ★ 当前最佳配置
│   ├── exp031_iterative.yaml      之前的迭代实验配置
│   └── ...
├── ic_models/
│   ├── ms_flow_pose_net.py       ★ 核心模型 (MSFlowPoseNet, ~676行)
│   ├── corr_pose_net.py           CorrPoseNet 基线模型
│   └── ic_pose_net*.py            早期模型 (v1/v2/v3, 已弃用)
├── modules/
│   ├── geometry_solver.py        ★ Image Jacobian + 加权最小二乘
│   ├── multiscale_renderer.py    ★ 多尺度3DGS渲染器 (gsplat)
│   ├── lie_algebra.py            ★ SE(3) 指数/对数映射
│   ├── conv_gru.py                ConvGRU 实现
│   └── ...
├── losses/
│   ├── pose_loss.py               PoseLoss (rotation + translation)
│   ├── sequence_loss.py           RAFT-style sequence loss
│   └── ...
├── data/
│   ├── dataset_v4.py             ★ 当前使用的数据集 (SD-Primary 多尺度特征)
│   └── dataset.py                 旧版数据集
├── feature_3dgs/
│   ├── gaussian_feature_model.py  GaussianFeatureModel (3DGS/2DGS)
│   ├── feature_renderer.py        gsplat 可微渲染 (rasterization/2dgs)
│   ├── train_feature_embedding.py 特征嵌入训练
│   └── ...
├── feature_extraction/
│   ├── extractor_sd.py            Stable Diffusion 多尺度特征提取
│   ├── extractor_dino.py          DINOv2 特征提取
│   └── fused_feature_extractor.py SD+DINO 融合提取
├── feature_compression/
│   ├── autoencoder.py             特征压缩自编码器
│   └── compressor.py              压缩器包装
├── utils/                         可视化、评估、度量等工具
└── output/                        训练输出 (模型、日志)
```

### 关键文件路径

| 功能 | 文件 |
|------|------|
| **训练入口** | `scripts/train_ms_flow.py` |
| **核心模型** | `ic_models/ms_flow_pose_net.py` |
| **几何求解** | `modules/geometry_solver.py` |
| **多尺度渲染** | `modules/multiscale_renderer.py` |
| **李代数** | `modules/lie_algebra.py` |
| **数据集** | `data/dataset_v4.py` |
| **Feature 3DGS** | `feature_3dgs/gaussian_feature_model.py` |
| **gsplat 渲染** | `feature_3dgs/feature_renderer.py` |
| **最佳配置** | `configs/exp032_cosine_fiters8.yaml` |

---

## 3. 环境配置

### 3.1 当前环境 (2×3090 服务器)

```
Python:     3.9.23
PyTorch:    1.13.1 + CUDA 11.6 + cuDNN 8.3.2
Conda env:  geo-aware
```

### ⚠️ 3.2 4090 适配注意事项

**RTX 4090 (Ada Lovelace, sm_89) 需要 CUDA 11.8+**，当前环境的 CUDA 11.6 不兼容！

推荐升级方案：

```bash
# 方案A: PyTorch 2.0+ (推荐，性能更好)
conda create -n geo-aware python=3.9
conda activate geo-aware
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 方案B: 保持 PyTorch 1.13 但升 CUDA 11.8
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3.3 核心依赖

```bash
# 基础
pip install numpy==1.23.5 opencv-python==4.8.1.78 pillow==9.5.0 scipy==1.13.1

# 3DGS 渲染
pip install gsplat==1.4.0  # 关键! 必须与CUDA版本匹配
# 注意: gsplat 1.4.0 需要从源码编译以匹配 CUDA 版本
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0

# 深度学习
pip install kornia==0.6.0 timm==0.6.11 einops==0.3.0
pip install transformers==4.26.1  # Stable Diffusion 特征提取
pip install lpips==0.1.4

# 3D处理
pip install open3d==0.17.0 plyfile==1.1.3

# 训练管理
pip install wandb==0.23.1 pyyaml tqdm

# Detectron2 (可选，部分旧代码依赖)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### 3.4 自编译包

以下包需要从源码编译:

```bash
# diff-gaussian-rasterization (高斯光栅化, 部分旧代码可能依赖)
cd submodules/diff-gaussian-rasterization
pip install .

# simple-knn (KNN查询)
cd submodules/simple-knn
pip install .
```

### 3.5 数据准备

```bash
# 确保以下目录结构存在:
output/
├── features_multiscale/room_0/     # 多尺度特征 (v1格式)
│   ├── coarse/                     # 7×10, 1280-d SD s5 特征
│   ├── mid/                        # 15×20, 1280-d SD s4 特征
│   ├── fine_sd/                    # 35×46, 640-d SD s3 特征
│   └── fine_dino/                  # 35×46, 768-d DINOv2 特征
├── 3dgs_models/room_0/
│   ├── coarse/point_cloud.ply      # 粗尺度 3DGS 模型
│   ├── mid/point_cloud.ply
│   ├── fine_sd/point_cloud.ply
│   └── fine_dino/point_cloud.ply
├── depth_maps/room_0/Sequence_*/   # 深度图 (PNG, uint16)
└── exp032_train.log                # 训练日志

dataset/Replica/room_0/
├── Sequence_1/
│   └── traj_w_c.txt                # 训练集 c2w 位姿
└── Sequence_2/
    └── traj_w_c.txt                # 验证集 c2w 位姿
```

---

## 4. 核心架构详解

### 4.1 整体流程

```
输入: 查询特征 {coarse, mid, fine_sd, fine_dino} + 初始位姿 T_init
       ↓
┌─── Outer Loop (3-5 次迭代) ────────────────────────────┐
│  1. 用当前位姿 T_curr 可微渲染参考特征图 (4个尺度)      │
│  2. MSFlowPoseNet 预测光流:                              │
│     Coarse (7×10): 全局相关性 → 粗光流                   │
│     Mid (15×20):   引导局部相关性 → 中光流                │
│     Fine (35×46):  RAFT-style GRU × 8 → 精细光流          │
│  3. Image Jacobian + WLS → 位姿增量 Δξ ∈ se(3)          │
│  4. T_curr = exp(Δξ) · T_curr                            │
└────────────────────────────────────────────────────────┘
       ↓
输出: 精化后的位姿 T_final
```

### 4.2 MSFlowPoseNet 内部结构 (`ic_models/ms_flow_pose_net.py`)

```python
class MSFlowPoseNet(nn.Module):
    # ~4.17M 可训练参数
    
    # 1. 特征解码器 (降维到64-d)
    ScaleDecoder:       高维 SD 特征 → 64-d, L2 归一化 (coarse/mid)
    FineDualDecoder:    SD s3 + DINO 分支 → 融合 → 64-d (fine)
    
    # 2. 相关性计算
    global_correlation():        全局 all-pairs (coarse, 7×10)
    guided_local_correlation():  warp 引导的局部 (mid/fine, r=4)
    
    # 3. 光流精化
    FlowRefinementHead:  corr_encoder → ConvGRU → flow_head
                         输出 (du, dv, confidence)
    
    # 4. 几何求解 (在 geometry_solver.py)
    diff_pose_solve():  加权最小二乘 + LM 阻尼 → 6-DOF Δξ
```

### 4.3 多尺度渲染 (`modules/multiscale_renderer.py`)

```python
class MultiScaleRenderer:
    # 4 个 GaussianFeatureModel 实例, 共享 Gaussian 几何
    models = {coarse, mid, fine_sd, fine_dino}
    
    render_batch():  # 批量可微渲染
        for scale in models:
            feats[scale] = gsplat.rasterization(...)  # 或 rasterization_2dgs
        depth = render_depth('D' mode, fine resolution)
        return feats, depth
```

### 4.4 几何求解 (`modules/geometry_solver.py`)

基于 **Image Jacobian** 的可微位姿估计:

$$J = \frac{\partial(u,v)}{\partial\xi} \in \mathbb{R}^{2 \times 6}$$

其中 $\xi = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]$ 是 se(3) 李代数。

加权最小二乘求解:
$$\Delta\xi = (J^T W J + \lambda I)^{-1} J^T W r$$

- $r$: 光流残差 (predicted flow)
- $W$: 置信度权重 (网络预测)
- $\lambda$: LM 阻尼项 (默认 1e-4)

### 4.5 训练损失

```python
# 1. 光流损失 (主损失)
multiscale_flow_loss():
    # Coarse/Mid: L1 loss
    # Fine (RAFT): γ^(N-1-i) 加权的 sequence loss (γ=0.85)
    # 有效像素掩码: 排除无深度/出界区域

# 2. 位姿损失 (Phase2 辅助)
pose_loss():
    # 旋转: cosine loss = 1 - cos(θ_error)
    # 平移: L2 norm of translation difference
    # 强制 fp32 计算, 避免 fp16 NaN
```

---

## 5. 训练流程

### 5.1 两阶段训练

| 阶段 | Epoch | 损失 | 说明 |
|------|-------|------|------|
| Phase1 | 1-3 | 仅 flow loss | 光流网络热身 |
| Phase2 | 4-60 | flow + pose loss | 端到端位姿优化 |

### 5.2 关键训练技术

**a) 外循环迭代精化 (Outer-loop)**
```yaml
outer_iters: 3      # 训练时每个样本迭代3次
val_outer_iters: 5   # 验证时迭代5次
```
每次迭代: 渲染→前向→几何求解→更新位姿→重新渲染

**b) 噪声课程学习 (Noise Curriculum)**
```yaml
noise_rot_min: 2.0    # 初始旋转噪声 (度)
noise_rot_max: 8.0    # 最终旋转噪声
noise_trans_min: 0.05  # 初始平移噪声 (米)
noise_trans_max: 0.2
warmup_epochs: 40      # 线性增长周期
```
从小噪声开始，逐步增大，让模型学会处理更大的位姿偏差

**c) Pose Loss 暖启动**
```yaml
pose_warmup_epochs: 8   # 前8个epoch逐步增加pose_weight
pose_warmup_min: 0.1    # 起始权重
pose_weight: 1.0         # 最终权重
trans_weight: 10.0        # 平移损失额外缩放
```

**d) Warmstart**
```yaml
warmstart: path/to/exp031/best.pth  # 加载预训练权重
```
exp032 从 exp031 最佳检查点启动，只加载模型权重，优化器从头开始

### 5.3 启动训练

```bash
cd /home/yons/Projects/ICLPose

# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py \
    --config configs/exp032_cosine_fiters8.yaml

# 多GPU (6×4090): 修改配置或使用 DDP
# 注意: 当前训练脚本是单GPU的, 需要改造为DDP
```

---

## 6. 实验历史与改进

### 6.1 演进时间线

| 实验 | 模型 | 关键改动 | 最佳 rot | <1° |
|------|------|---------|---------|-----|
| exp001-009 | ICPoseNet v1/v2 | Transformer 回归 | ~5-10° | <50% |
| exp010-014 | CorrPoseNet | 光流 + 几何求解 | ~1-2° | ~70% |
| **exp015** | **CorrPoseNet** | **调参完善** | **0.46°** | **83.6%** |
| exp029-030 | MSFlowPoseNet | 单 pass, 无外循环 | >>5° | 极差 |
| exp031 | MSFlowPoseNet | +外循环迭代, acos rot loss | 0.87° (bug) | **95.6%** |
| **exp032** | **MSFlowPoseNet** | **+cosine loss, fiters8, warmstart** | **0.33°** | **95.6%** |

### 6.2 关键 Bug 修复

#### Bug 1: 0.81° 指标地板 (exp031)
```python
# 原因: acos(1 - clamp_value) = acos(1 - 1e-4) = 0.8103°
cos_theta = cos_theta.clamp(-1+1e-4, 1-1e-4)  # ❌ clamp 太松
# 修复:
cos_theta = cos_theta.clamp(-1+1e-7, 1-1e-7)  # ✅ 精度足够
```
exp031 卡在 0.81° 是因为指标计算的 clamp 值太大，即使预测完美，显示也不会低于 0.81°。

#### Bug 2: acos 梯度爆炸
```python
# acos'(x) = -1/√(1-x²), 当 x→±1 时梯度→∞
# 解决方案: 改用 cosine loss (1-cos(θ)), 梯度有界
loss_rot = 1 - cos_theta  # ✅ gradient-safe
```

#### Bug 3: FP16 NaN
```python
# Pose loss 中的小数值运算在 fp16 下会产生 NaN
# 解决: 强制 fp32
with torch.cuda.amp.autocast(enabled=False):
    loss_rot = compute_rotation_loss(pred.float(), gt.float())
```

### 6.3 exp031 → exp032 关键变化

| 参数 | exp031 | exp032 |
|------|--------|--------|
| rot_loss_type | acos | **cosine** |
| fine_iters | 4 | **8** |
| lr | 1e-4 | **5e-5** |
| phase1_epochs | 15 | **3** |
| warmup_epochs | 25 | **40** |
| noise_rot_min | 5.0 | **2.0** |
| metric clamp | 1e-4 | **1e-7** |
| warmstart | None | **exp031 best.pth** |
| trans_weight | 1.0 | **10.0** |

---

## 7. 当前最佳结果

### exp032 (E16, 当前最佳)

| 指标 | exp032 E16 | CorrPoseNet 基线 | 改进 |
|------|-----------|-----------------|------|
| **Rot Mean** | **0.33°** | 0.46° | **-28%** ✅ |
| Rot Median | 0.23° | — | — |
| **<1°** | **95.6%** (E15) | 83.6% | **+12%** ✅ |
| Trans Mean | 20.7mm | 25.5mm | -19% ✅ |
| Flow EPE | 0.20 | — | — |

### 训练状态 (截止 E16)

- exp032 正在 GPU1 继续训练，当前 E16/60
- 噪声课程: rot=4.5°, trans=0.135m (ratio 42%)
- 学习率: 5e-5 (恒定)

### exp031 对比

exp031 因 0.81° 指标 bug，始终显示 rot≈0.87°，但 <1° 达到了 95.6%。
实际上 exp031 的真实旋转精度可能在 0.4-0.5° 左右（因为 <1° 和 CorrPoseNet 相当甚至更好）。

---

## 8. 后续优化方向

### 8.1 高优先级

1. **多场景泛化**: 只在 room_0 训练过，需要测试 room_1, room_2 等
2. **多GPU训练**: 当前单GPU，6×4090 应实现 DDP/FSDP 加速
3. **NetVLAD 初始化**: 验证时使用更真实的初始位姿（如 NetVLAD 检索）替代随机扰动
4. **大噪声鲁棒性**: 继续训练到 warmup 结束 (E40)，看大噪声下是否还能收敛

### 8.2 中优先级

5. **2DGS 支持**: 代码已支持 2DGS surfel 渲染，但训练流程未验证
6. **更大 Fine Iters**: 当前 fine_iters=8，可尝试 12 或 16
7. **更深 GRU**: 当前 1 层 ConvGRU + 2×conv flow head，可增加深度
8. **Online feature extraction**: 当前是离线提取特征，在线提取可减少存储

### 8.3 低优先级

9. **混合精度优化**: 当前 AMP 已启用但 pose loss 被强制 fp32，可优化
10. **新场景 (OldHospital 等)**: 扩展到更复杂的室外场景

---

## 9. 快速上手

### 9.1 仅运行推理/验证

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py \
    --config configs/exp032_cosine_fiters8.yaml \
    --eval_only \
    --resume output/exp032/best.pth
```

### 9.2 继续训练 (从 exp032 检查点)

```yaml
# 创建新配置 configs/exp033_xxx.yaml, 关键字段:
experiment_name: exp033_xxx
warmstart: output/exp032/best.pth
# 调整其他超参...
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py \
    --config configs/exp033_xxx.yaml
```

### 9.3 核心文件修改指引

- **改模型结构**: `ic_models/ms_flow_pose_net.py` 中的 `MSFlowPoseNet`
- **改损失函数**: `scripts/train_ms_flow.py` 中的 `multiscale_flow_loss()` 和 `pose_loss()`
- **改训练循环**: `scripts/train_ms_flow.py` 中的 `MSFlowTrainer._train_step()`
- **改几何求解**: `modules/geometry_solver.py` 中的 `diff_pose_solve()`
- **改渲染逻辑**: `modules/multiscale_renderer.py` 中的 `render_batch()`
- **改数据加载**: `data/dataset_v4.py` 中的 `PoseDatasetV4`

### 9.4 实验管理

训练日志保存在 `output/{exp_name}_train.log`，可用以下命令监控:

```bash
# 查看最新验证结果
grep '\[Val E' output/exp032_train.log | tail -5

# 查看最佳指标
grep '★' output/exp032_train.log

# 查看噪声课程
grep 'Curriculum' output/exp032_train.log | tail -5
```

Wandb 集成已配置，训练时自动上传指标到 wandb.ai。

---

## 10. OldHospital 2DGS 几何重建 (另一条工作线)

> **⚠️ 重要**: 本项目有两条并行的工作线。上面所有章节覆盖的是**定位网络** (MSFlowPoseNet)。
> 另一条同等重要的工作线是 **OldHospital 场景的 2DGS 几何重建 + WildGaussians 外观建模**。

### 10.1 概述

- 目标: 将 OldHospital 2DGS 重建 PSNR 提升到 ≥20 dB
- 当前最佳: **18.21 dB** (retrain38f, 100-step test-time opt)
- 核心脚本: `feature_3dgs/train_2dgs_geometry.py` (~2700 行)
- 数据集: `dataset/OldHospital/` (~32 GB)

### 10.2 完整文档

**请参阅 `docs/OLDHOSPITAL_2DGS_GUIDE.md`** — 包含完整的:
- 数据集描述、WildGaussians 架构、实验历史 (retrain31→38g)
- CLI 参数、训练配置、Eval-only 模式
- 关键发现/教训、继续优化方向

### 10.3 数据迁移 (补充原有迁移步骤)

```bash
# OldHospital 数据集 (~15 GB 必需)
rsync -avz dataset/OldHospital/ new_server:ICLPose/dataset/OldHospital/

# 2DGS 最佳检查点 (~400 MB)
rsync -avz output/2dgs_models/OldHospital/v3_retrain38f/point_cloud/iteration_30000/ \
    new_server:ICLPose/output/2dgs_models/OldHospital/v3_retrain38f/point_cloud/iteration_30000/
```

### 10.4 快速启动 2DGS 训练

```bash
# 在 4090 上训练 (单卡 ~20 GB 显存)
CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain39 \
    --iterations 30000 --batch_size 4 --use_mask --random_background \
    --use_appearance --wildgaussians \
    --wg_output_scale 0.5 --wg_hidden_dim 256 --wg_n_hidden 3 \
    --wg_image_embed_dim 64 --no_dino_uncertainty \
    --lambda_dssim 0.2 --lambda_normal 0 --lambda_dist 0 --lambda_scale 0 --lambda_depth 0 \
    --densify_until_iter 15000 --densify_grad_threshold 0.0002 --max_gaussians 200000 \
    --opacity_reset_interval 30001 --appearance_reg 0.001 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 15000 30000 --wg_test_opt_steps 0

# 完整评估 (100-step test-time opt)
CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain39 \
    --use_mask --wildgaussians --eval_only --checkpoint_iter 30000 --wg_test_opt_steps 100
```

---

## 附录 A: 完整 conda 包列表

参见 `docs/ENVIRONMENT_SETUP.md` 中的完整环境导出。

## 附录 B: Git 提交历史

```
9e9a325 feat/fix: exp030-032 updates (iterative routing, config tuning, fine iters)
Branch: v2-iterative-routing
Remote: git@github-sqy:Arthurshen926/ICLPose.git
```
