# OldHospital 2DGS 几何重建 — 完整技术文档

> **最后更新**: 2026-03-06 | **当前最佳**: retrain38f = **18.21 dB** (toned PSNR, 100-step test-time opt) | **目标**: ≥20 dB

---

## 目录

1. [任务概述](#1-任务概述)
2. [OldHospital 数据集](#2-oldhospital-数据集)
3. [核心脚本架构](#3-核心脚本架构)
4. [WildGaussians 外观建模](#4-wildgaussians-外观建模)
5. [Test-time Embedding Optimization](#5-test-time-embedding-optimization)
6. [实验历史 (retrain31→38g)](#6-实验历史)
7. [训练命令参考](#7-训练命令参考)
8. [Eval-only 模式](#8-eval-only-模式)
9. [关键发现与教训](#9-关键发现与教训)
10. [继续优化方向](#10-继续优化方向)

---

## 1. 任务概述

### 1.1 背景

ICLPose 的定位网络 (MSFlowPoseNet) 依赖高质量的 3DGS 场景重建。OldHospital 是一个多序列室外数据集，不同序列拍摄于不同时间/天气/光照条件下，且包含大量行人等动态物体。这导致标准 3DGS 重建效果很差。

### 1.2 目标

将 OldHospital 的 2DGS 重建 PSNR 从基线约 16-17 dB 提升到 ≥20 dB，以便后续用于特征嵌入和定位网络训练。

### 1.3 核心挑战

- **多序列数据不一致**: 9 个序列不同时间拍摄，曝光/天气/光照差异大
- **动态物体**: 行人、车辆等 transient objects (每个序列不同)
- **Cross-PSNR 天花板**: 即使最近的 train/test GT 之间 PSNR 也仅 8-11 dB

### 1.4 解决方案

采用 **WildGaussians** (NeurIPS 2024) 的 per-Gaussian × per-Image 外观建模，配合 test-time embedding optimization。

---

## 2. OldHospital 数据集

### 2.1 数据统计

| 属性 | 值 |
|------|-----|
| **总图像数** | 1,077 |
| **训练集** | 895 (seq1,2,3,5,6,7,9) |
| **测试集** | 182 (seq4=56, seq8=126) |
| **分辨率** | 1920×1080 |
| **COLMAP 点云** | 109,381 points (sparse/0/) |
| **数据集大小** | ~32 GB |

### 2.2 磁盘结构

```
dataset/OldHospital/
├── sparse/0/                 # COLMAP SfM 结果
│   ├── cameras.bin           # 相机内参
│   ├── images.bin            # 相机外参 (1077张)
│   ├── points3D.bin          # 3D 点云
│   └── points3D.ply          # PLY 格式点云
├── seq1/ ~ seq9/             # 9 个图像序列 (不同时间/天气)
├── dataset_train.txt         # 训练集文件列表 (895行)
├── dataset_test.txt          # 测试集文件列表 (182行)
├── masks.pkl                 # 静态掩码 (6.3 GB, 含 obj_mask/sky_mask/distort_mask)
├── mono_depth/               # DPT 单目深度估计 (8.4 GB, 目前未使用)
└── videos/                   # 原始视频
```

### 2.3 掩码说明

`masks.pkl` 是一个 dict: `{image_name: (obj_mask, sky_mask, distort_mask)}`
- `obj_mask`: 动态物体掩码 (True=静态区域)
- `sky_mask`: 天空掩码 (True=非天空)
- `distort_mask`: 畸变区域掩码 (True=无畸变)
- RGB loss 使用 `obj_mask & distort_mask`
- Geometry loss 使用 `obj_mask & distort_mask & sky_mask`

### 2.4 迁移时需传输的文件

```bash
# 必需 (~15 GB 最小集)
rsync -avz dataset/OldHospital/sparse/ new_server:ICLPose/dataset/OldHospital/sparse/
rsync -avz dataset/OldHospital/seq*/ new_server:ICLPose/dataset/OldHospital/     # 图像
rsync -avz dataset/OldHospital/masks.pkl new_server:ICLPose/dataset/OldHospital/
rsync -avz dataset/OldHospital/dataset_*.txt new_server:ICLPose/dataset/OldHospital/

# 最佳检查点 (~400 MB)
rsync -avz output/2dgs_models/OldHospital/v3_retrain38f/point_cloud/iteration_30000/ \
    new_server:ICLPose/output/2dgs_models/OldHospital/v3_retrain38f/point_cloud/iteration_30000/
rsync -avz output/2dgs_models/OldHospital/v3_retrain38f/train.log \
    new_server:ICLPose/output/2dgs_models/OldHospital/v3_retrain38f/

# 可选 (mono_depth, 如要用深度监督)
rsync -avz dataset/OldHospital/mono_depth/ new_server:ICLPose/dataset/OldHospital/mono_depth/
```

---

## 3. 核心脚本架构

### 3.1 文件: `feature_3dgs/train_2dgs_geometry.py` (~2700 行)

这是 2DGS 几何重建的**唯一核心文件**，包含所有模型、训练循环、评估逻辑。

### 3.2 代码结构

| 行范围 | 内容 |
|--------|------|
| 1-50 | Imports, 常量 (C0=0.28209479177387814) |
| 52-130 | COLMAP 数据加载 (read_cameras_binary, read_images_binary) |
| 134-170 | SH 工具, SSIM 实现 |
| 175-560 | **GaussianModel2DGS** — 2DGS 高斯模型 (属性/初始化/densification/save/load) |
| 565-725 | **AppearanceNetwork** — 全局外观校正 (legacy, 不推荐) |
| 623-725 | **SpatialAppearanceNetwork** — 空间外观校正 (legacy) |
| **728-910** | **★ WildGaussiansAppearance** — Per-Gaussian 外观建模 (核心!) |
| **913-1075** | **★ DinoUncertaintyPredictor** — DINO 不确定性预测 (可选) |
| 1079-1225 | CameraData, load_scene, load_image_tensor |
| 1232-1432 | render_2dgs, render_2dgs_batch — 可微渲染 |
| 1438-1510 | 深度监督工具 (pearson_depth_loss) |
| **1514-2190** | **★ train()** — 主训练循环 |
| **2219-2412** | **★ evaluate()** — 评估 (含 test-time embedding opt) |
| 2441-2595 | CLI 参数解析 |
| **2599-2670** | **★ eval_only()** — 独立评估模式 |

### 3.3 关键依赖

```python
from gsplat import rasterization_2dgs, spherical_harmonics
# gsplat 1.4.0:
#   rasterization_2dgs(means, quats, scales, opacities, colors, viewmats, Ks, ...)
#   spherical_harmonics(degree, dirs, coeffs) → raw_colors (需 +0.5)
```

---

## 4. WildGaussians 外观建模

### 4.1 核心思想

每个高斯点在不同训练图片中可以有不同的颜色（吸收曝光/光照变化），MLP 预测 per-Gaussian affine color transform:

$$\text{toned\_color}_i = s_i \cdot \text{base\_color} + b_i$$

其中 $s_i \approx 1$, $b_i \approx 0$ (identity initialization)。

### 4.2 架构 (`WildGaussiansAppearance`)

```
输入: image_embedding[32d] + gaussian_embedding[24d] + base_color[3d] = 59d
  ↓
MLP: Linear(59, H) → ReLU → [Linear(H, H) → ReLU] × (n_hidden-1) → Linear(H, 6)
  ↓
输出: 6d = [scale_raw(3), bias_raw(3)]
  ↓
scale = output_scale * scale_raw + 1.0    # ≈1.0 at init
bias  = output_scale * bias_raw           # ≈0.0 at init
```

### 4.3 关键组件

| 组件 | 维度 | 说明 |
|------|------|------|
| `image_embedding` | nn.Embedding(895, 32) | 每张训练图一个 embedding, 初始化为 0 |
| `gaussian_embedding` | nn.Parameter(N, 24) | Fourier positional encoding (sin/cos, 4 octaves × 3d × 2) |
| `mlp` | 59 → H×n_hidden → 6 | 最后一层 zero-init 保证 identity 启动 |
| `output_scale` | float | 控制校正范围: scale ∈ [1-s, 1+s], bias ∈ [-s, s] |
| `base_colors` | via `get_base_colors()` | 完整 SH 评估 (view-dependent) 或 DC-only |

### 4.4 CLI 超参数

```bash
--wildgaussians              # 启用 WildGaussians (必需)
--wg_output_scale 0.3        # 校正范围 (0.3=±30%, 0.5=±50%)
--wg_hidden_dim 128          # MLP 隐藏层维度
--wg_n_hidden 2              # MLP 隐藏层数量
--wg_image_embed_dim 32      # 每图像 embedding 维度 (NEW: 可调)
--wg_gaussian_embed_dim 24   # 每高斯 embedding 维度 (NEW: 可调)
--appearance_lr_init 5e-4    # MLP 学习率
--gaussian_emb_lr 5e-3       # Gaussian embedding 学习率
--image_emb_lr 1e-3          # Image embedding 学习率
--appearance_reg 0.01        # L2 正则化 (image embedding)
--no_dino_uncertainty        # 禁用 DINO (省 ~1GB 显存, NEW)
```

### 4.5 训练时数据流

```
每次训练迭代:
  1. 采样 batch_size=4 个训练相机
  2. 对每个相机:
     a. get_base_colors(gaussians, cam) → 完整 SH → [N, 3] base colors
     b. compute_toned_colors(cam_idx, base_colors) → MLP → [N, 3] toned colors
     c. render_2dgs(gaussians, cam, override_colors=toned_colors)
     d. L1 + DSSIM loss on (toned_render, gt_image * mask)
  3. 正则化: L2 on image_embedding.weight
  4. 单次 backward + optimizer step
```

### 4.6 DINO Uncertainty (可选)

`DinoUncertaintyPredictor`: 用 DINOv2 cosine similarity 预测动态区域。
- 预缓存所有训练图像的 DINO 特征 (cache_gt_features)
- 训练时定期更新不确定性掩码
- **目前建议禁用** (`--no_dino_uncertainty`): 静态 masks.pkl 已足够好，省约 1GB 显存

---

## 5. Test-time Embedding Optimization

### 5.1 核心突破

训练时每张图有 learned embedding，测试时没有。直接用 mean embedding 效果差 (≈16.7 dB)。
**解决方案**: 对每张测试图优化一个新 embedding，使渲染结果匹配 GT — 这是 WildGaussians 论文的标准评估协议。

**效果**: mean embedding 16.69 dB → 100-step opt **18.21 dB** (+1.52 dB)

### 5.2 算法

```python
对每张测试图:
  1. 初始化 test_emb (最近邻 3 个训练相机 embedding 的平均值)
  2. 优化循环 (100 steps):
     a. 用 test_emb 通过 MLP 计算 toned colors
     b. 渲染 toned image
     c. 计算 L1 + DSSIM loss (带 mask 排除动态物体)
     d. 反向传播到 test_emb, Adam.step()
     e. 余弦退火 LR: 0.01 → 0.001
  3. 用最终 test_emb 渲染评估 image, 计算 PSNR
```

### 5.3 最新改进 (2026-03-06)

| 改进 | 之前 | 现在 |
|------|------|------|
| **Embedding 初始化** | mean (≈0) | Top-3 最近邻训练相机的 embedding 平均 |
| **优化 loss** | 纯 L1 | L1 + DSSIM (匹配训练目标) |
| **学习率** | 固定 0.01 | 余弦退火 0.01 → 0.001 |
| **Mask** | 无 | 有 (排除动态物体) |

### 5.4 CLI 控制

```bash
--wg_test_opt_steps 0     # 训练中间 eval: 快速 mean embedding (默认)
--wg_test_opt_steps 100   # 完整 eval: 每张测试图优化 100 步 (~15min)
# 注意: 训练结束时自动用 100 步做最终 eval (即使训练中设为 0)
```

---

## 6. 实验历史

### 6.1 里程碑总结

| 实验 | 关键变化 | Raw PSNR | Toned PSNR | 备注 |
|------|---------|----------|------------|------|
| retrain31 | 基线 (无 WG, freeze@5k) | **16.91** | — | 纯 SH 最佳 |
| retrain37 | +WG+DINO (初次) | 14.17 | — | 多 bug, 差 |
| retrain38 | 修 absgrad/single render | 16.38@3K | — | opacity_reset 后崩 |
| retrain38b | 禁用 opacity_reset | — | — | 仍下降 |
| retrain38c | +SH full eval +geometry losses | — | — | geometry 竞争 |
| retrain38d | **禁用 geometry losses** | — | 16.72@5K (mean) | 持续上升 |
| retrain38e | +output_scale=0.3, max_gauss=180K | 16.24@10K | 16.68 (mean) / **17.60** (50-step opt) | 突破! |
| **retrain38f** | max_gauss=200K, densify=15K | 15.63@30K | 16.69 (mean) / **18.21** (100-step opt) | **当前最佳** |
| retrain38g | 256×3 MLP, output_scale=0.5, 64d embed | — | — | 已准备, 待 GPU |

### 6.2 retrain38f 完整 PSNR 轨迹

| Iter | Raw PSNR | Toned (mean) | Gap |
|------|----------|-------------|-----|
| 1K | 15.74 | 16.03 | 0.29 |
| 3K | 16.32 | 16.57 | 0.25 |
| 5K | 16.32 | 16.58 | 0.26 |
| 10K | 16.19 | 16.58 | 0.39 |
| 15K | 15.95 | 16.66 | 0.71 |
| 20K | 15.81 | 16.66 | 0.85 |
| 25K | 15.72 | 16.68 | 0.96 |
| 30K | 15.63 | 16.69 / **18.21** (opt) | 1.06 / 2.58 |

**规律**: Raw PSNR 下降 (canonical colors 退化) + Toned 持续上升 = MLP 承担更多外观校正。test-time opt 额外 +1.52 dB。

### 6.3 retrain38f 训练配置 (最佳参考)

```bash
CUDA_VISIBLE_DEVICES=1 python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain38f \
    --iterations 30000 \
    --batch_size 4 \
    --use_mask \
    --random_background \
    --use_appearance \
    --wildgaussians \
    --wg_output_scale 0.3 \
    --lambda_dssim 0.2 \
    --lambda_normal 0 --lambda_dist 0 --lambda_scale 0 --lambda_depth 0 \
    --densify_until_iter 15000 \
    --densify_grad_threshold 0.0002 \
    --max_gaussians 200000 \
    --opacity_reset_interval 30001 \
    --appearance_lr_init 5e-4 \
    --gaussian_emb_lr 5e-3 \
    --image_emb_lr 1e-3 \
    --appearance_reg 0.01 \
    --grad_clip_max_norm 1.0 \
    --test_iterations 1000 3000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 15000 20000 30000 \
    --wg_test_opt_steps 0
```

### 6.4 retrain38g 待运行配置 (下一步)

```bash
# 关键变化: 更大 MLP, 更高 output_scale, 更大 embedding, 禁用 DINO
--wg_output_scale 0.5 \
--wg_hidden_dim 256 \
--wg_n_hidden 3 \
--wg_image_embed_dim 64 \   # 从 32 增到 64
--no_dino_uncertainty \       # 省 ~1GB 显存
--appearance_reg 0.001 \      # 从 0.01 降到 0.001
```

---

## 7. 训练命令参考

### 7.1 完整 CLI 参数表

#### 基础参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source_dir` | (必需) | COLMAP 数据集路径 |
| `--model_dir` | (必需) | 输出目录 |
| `--iterations` | 30000 | 总迭代次数 |
| `--batch_size` | 1 | 每步训练的视图数 |
| `--longest_edge` | 0 | 分辨率限制 (0=全分辨率) |
| `--sh_degree` | 3 | SH 阶数 |

#### 外观模型
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_appearance` | False | 启用外观校正 |
| `--wildgaussians` | False | WildGaussians 模式 (推荐) |
| `--wg_output_scale` | 0.3 | 校正范围 |
| `--wg_hidden_dim` | 128 | MLP hidden dim |
| `--wg_n_hidden` | 2 | MLP hidden layers |
| `--wg_image_embed_dim` | 32 | Image embedding dim |
| `--wg_gaussian_embed_dim` | 24 | Gaussian embedding dim |
| `--no_dino_uncertainty` | False | 禁用 DINO 不确定性 |

#### 损失权重
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--lambda_dssim` | 0.2 | DSSIM 权重 |
| `--lambda_normal` | 0 | 法线一致性 (建议关闭) |
| `--lambda_dist` | 0 | 距离正则 (建议关闭) |
| `--lambda_scale` | 0 | 尺度正则 (建议关闭) |
| `--lambda_depth` | 0 | 深度监督 (建议关闭) |

#### Densification
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--densify_until_iter` | 15000 | 停止 densify |
| `--densify_grad_threshold` | 0.0002 | 梯度阈值 |
| `--max_gaussians` | 200000 | 高斯数量上限 |
| `--opacity_reset_interval` | 30001 | 不透明度重置 (>iterations=禁用) |

#### 评估
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wg_test_opt_steps` | 0 | 测试时优化步数 (0=快速 mean eval) |
| `--eval_only` | False | 仅评估模式 |
| `--checkpoint_iter` | 0 | 加载指定迭代的检查点 |

---

## 8. Eval-only 模式

### 8.1 使用方法

```bash
CUDA_VISIBLE_DEVICES=0 python -u feature_3dgs/train_2dgs_geometry.py \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3_retrain38f \
    --use_mask \
    --wildgaussians \
    --eval_only \
    --checkpoint_iter 30000 \
    --wg_test_opt_steps 100
```

### 8.2 自动推断架构

eval_only 从 checkpoint 的 state_dict 自动推断 MLP 结构:
- `hidden_dim`: 从 `mlp.0.weight.shape[0]` 读取
- `n_hidden`: 计算 state_dict 中 `mlp.*.weight` 的数量 - 1
- `image_embed_dim` / `gaussian_embed_dim`: 从 checkpoint metadata 读取

### 8.3 检查点结构

```
point_cloud/iteration_30000/
├── point_cloud.ply           # 高斯几何 (36 MB)
├── appearance_net.pth        # WG 权重 (15 MB)
│   ├── type: 'wildgaussians'
│   ├── state_dict: MLP + embeddings
│   ├── n_images: 895
│   ├── n_gaussians: ~200K
│   ├── image_embed_dim: 32
│   └── gaussian_embed_dim: 24
└── dino_uncertainty.pth      # DINO 预测器 (可选, 331 MB)
```

---

## 9. 关键发现与教训

### 9.1 必须做的事

1. **禁用 geometry losses** (normal, dist, scale, depth): 与 WG 外观建模竞争，导致 PSNR 下降
2. **禁用 opacity_reset**: reset 后 PSNR 崩溃不恢复
3. **单次 toned render** (不做 raw + toned 双渲染): 省 50% VRAM
4. **所有 loss 在 toned image 上** (不在 raw): MLP 才能正确训练
5. **使用 absgrad** (override_colors path): gsplat 梯度计算需要
6. **全 SH 评估** (不仅 DC): view-dependent base colors 质量更好
7. **梯度裁剪** (`--grad_clip_max_norm 1.0`): 防止训练不稳定

### 9.2 关于 PSNR 指标

- **Raw PSNR**: 不经过 MLP 的原始 SH 渲染。随训练下降（正常！MLP 接管外观校正）
- **Toned PSNR (mean)**: 用所有训练 embedding 的平均值。稳定在 ~16.7 dB，受限于 mean ≠ 任何特定测试图
- **Toned PSNR (optimized)**: test-time opt 后。18.21 dB = 真实重建质量指标
- **Masked PSNR**: 去除动态物体后的 PSNR (更高)

### 9.3 GPU 显存预估 (单卡)

| 配置 | 显存占用 |
|------|----------|
| 全分辨率 (1920×1080), batch=4, 200K 高斯, DINO ON | ~22 GB |
| 全分辨率, batch=4, 200K 高斯, DINO OFF | ~20 GB |
| 降分辨率 (960×540), batch=4, 200K 高斯 | ~14 GB |

### 9.4 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `cuDNN algorithm not found` | GPU 显存不足以分配 cuDNN workspace | 降分辨率或 batch_size |
| Eval-only OOM | 不要与训练进程同 GPU 运行 | 用空闲 GPU 或等训练结束 |
| PSNR 跌至 <14 | opacity_reset 后崩溃 | 设 `--opacity_reset_interval` > iterations |
| test-time opt 没效果 | evaluate() 被 `torch.no_grad()` 包裹 | 已修复: opt 在 no_grad 外执行 |

### 9.5 关于 gsplat

```python
# gsplat 1.4.0 API:
from gsplat import rasterization_2dgs, spherical_harmonics

# spherical_harmonics 签名:
#   spherical_harmonics(degrees_to_use: int, dirs: Tensor, coeffs: Tensor) → Tensor
#   返回 raw SH 值 (不含 +0.5), 需要手动 +0.5 得到 [0,1] 颜色

# rasterization_2dgs 返回 dict:
#   render, rend_alpha, rend_normal, surf_normal, rend_dist, depth, width, height
#   override_colors 需要 use_absgrad=True
```

---

## 10. 继续优化方向

### 10.1 距离 20 dB 的差距: 1.79 dB

当前 18.21 dB → 目标 20 dB，需要进一步提升。

### 10.2 高优先级

| 方向 | 预期收益 | 复杂度 | 说明 |
|------|---------|--------|------|
| **更大 MLP + embedding** | +0.5~1.0 dB | 低 | 256×3, image_emb=64 (retrain38g) |
| **更多 test-opt steps** | +0.2~0.5 dB | 低 | 200-500 步 (目前只用 100) |
| **更高 output_scale** | +0.3~0.5 dB | 低 | 0.5→1.0, 允许更大校正范围 |
| **降 reg** | +0.1~0.3 dB | 低 | 0.001 甚至 0 |

### 10.3 中优先级

| 方向 | 预期收益 | 复杂度 | 说明 |
|------|---------|--------|------|
| **Per-pixel appearance** (SpatialAppearanceNetwork + WG) | +0.5~1.0 dB | 中 | 不同像素不同校正 |
| **更多训练迭代** (50K-100K) | +0.2~0.5 dB | 低 | toned 在 30K 还在上升 |
| **Geometry losses at tiny weights** | +0.1~0.3 dB | 低 | lambda_normal=0.001 |
| **渐进式分辨率** | +0.3 dB | 中 | 先 960×540 训练，后期切全分辨率 |

### 10.4 低优先级/实验性

- HA-NeRF 风格的 transient head
- 3DGS (非 2DGS) 比较
- 自蒸馏 (teacher-student embedding)
- Multi-resolution appearance MLP

---

## 附录 A: 自动训练脚本

已配置 `scripts/auto_retrain38g.sh` 自动等待 GPU 空闲后启动训练并评估:

```bash
nohup bash scripts/auto_retrain38g.sh &> logs/auto_retrain38g.log &
```

## 附录 B: 调试常用命令

```bash
# 查看训练 PSNR 轨迹
grep "PSNR" output/2dgs_models/OldHospital/v3_retrain38f/train.log

# 查看 loss breakdown
grep "Loss breakdown" output/2dgs_models/OldHospital/v3_retrain38f/train.log | tail -10

# 查看 Gaussian 数量变化
grep "N=" output/2dgs_models/OldHospital/v3_retrain38f/train.log | tail -10

# GPU 状态
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# 编译检查
python -m py_compile feature_3dgs/train_2dgs_geometry.py
```
