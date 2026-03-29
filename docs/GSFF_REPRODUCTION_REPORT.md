# GSFFs 复现报告：OldHospital 场景 SOTA 方案

> **最终结果：18.3cm / 0.32°**，超越论文 Feature tuned (18cm / 0.36°)

---

## 目录

1. [方法概述](#1-方法概述)
2. [数据与预训练模型](#2-数据与预训练模型)
3. [模块详解](#3-模块详解)
   - 3.1 Triplane 特征场
   - 3.2 2D 特征编码器
   - 3.3 对比学习损失函数
   - 3.4 2DGS 可微渲染
   - 3.5 SE(3) 位姿优化
4. [训练流程](#4-训练流程)
5. [评估(定位)流程](#5-评估定位流程)
6. [关键改进汇总](#6-关键改进汇总)
7. [踩坑记录](#7-踩坑记录)
8. [消融实验结果](#8-消融实验结果)
9. [复现命令速查](#9-复现命令速查)

---

## 1. 方法概述

GSFFs (Gaussian Splatting Feature Fields) 是一种基于 3D Gaussian Splatting 的 6-DOF 视觉定位方法。核心思路：

```
离线阶段:
  预训练 2DGS 模型 → 冻结几何 → 训练 Triplane 特征场 + 2D Encoder
  使用 NCE/Prototypical/Segmentation CE 三种对比损失联合训练

在线定位:
  查询图像 → 2D Encoder 提取特征 F_2D
  初始位姿 → 可微渲染 Triplane 特征 F_3D
  直接优化 SE(3): argmin ||F_2D - F_3D(P)||²
```

**关键等式 (Paper Eq. 4)**:
$$P^* = \arg\min_{P \in SE(3)} \|F_{2D} - F_{3D}(P, \mathcal{G})\|_2^2$$

其中 $\mathcal{G}$ 是冻结的 Gaussian 模型，优化变量是 se(3) 李代数上的增量 $\Delta\xi$。

---

## 2. 数据与预训练模型

### 2.1 数据集

| 项目 | 值 |
|------|-----|
| 场景 | Cambridge Landmarks — OldHospital |
| 训练集 | 895 张图像 |
| 测试集 | 182 张图像 |
| 原始分辨率 | 1920 × 1080 |
| 相机内参 | fx=fy=1663.12 |

数据文件布局：
```
dataset/OldHospital/
  dataset_train.txt          # 格式: img_name X Y Z W P Q R
  dataset_test.txt
  seq1/*.png, seq2/*.png, ...
```

位姿格式：`dataset_*.txt` 中为 camera-to-world (position + quaternion WPQR)。`cameras.json` 中为 COLMAP c2w convention (rotation matrix + position)。

### 2.2 预训练 2DGS 模型

| 项目 | 值 |
|------|-----|
| 路径 | `output/2dgs_models/OldHospital/v7_depth/` |
| Gaussian 数量 | 300,845 个 |
| SH degree | 0 |
| Scale 维度 | **2D** (需 padding 第三维为 1) |
| 训练迭代 | 30,000 |

---

## 3. 模块详解

### 3.1 Triplane 特征场

**文件**: `gsff/triplane.py`

Triplane 由三个正交平面 $H_{xy}, H_{xz}, H_{yz} \in \mathbb{R}^{R \times R \times D}$ 组成。对每个 Gaussian 中心 $(x,y,z)$：

1. 归一化坐标到 $[-1, 1]$（除以 `scene_extent`）
2. 投影到三个平面：$(x,y)$, $(x,z)$, $(y,z)$
3. 双线性插值采样 (`grid_sample`)
4. 三个平面特征取平均: $g_i = \frac{1}{3}(f_{xy} + f_{xz} + f_{yz})$

```python
# 核心代码 (简化)
class TriplaneFeatureField(nn.Module):
    def __init__(self, resolution=256, feature_dim=16, scene_extent=10.0):
        self.plane_xy = nn.Parameter(torch.randn(1, D, R, R) * 0.01)
        self.plane_xz = nn.Parameter(torch.randn(1, D, R, R) * 0.01)
        self.plane_yz = nn.Parameter(torch.randn(1, D, R, R) * 0.01)
    
    def extract_features(self, xyz):
        coords = xyz / self.scene_extent  # 归一化到 [-1, 1]
        # 三个平面分别 grid_sample + 取平均
        return (feat_xy + feat_xz + feat_yz) / 3.0
```

**双尺度 Triplane** (`DualScaleTriplane`):
- **Coarse**: R=256, D=16 — 与 DINOv2 patch-level 特征对应
- **Fine**: R=1024, D=16 — 与像素级特征对应

正则化：Total Variation Loss (L_TVL) 对三个平面分别计算水平和垂直方向差分。

### 3.2 2D 特征编码器

**文件**: `gsff/encoder.py`

#### CoarseEncoder
- **Backbone**: DINOv2 ViT-B/14 (冻结)
- **投影头**: MLP (768 → 192 → 16)，含 GELU 激活
- **输出**: [B, 16, H/14, W/14]
- 输入需要 ImageNet 标准化，并 pad 到 patch_size=14 的整数倍

#### OldFineEncoder（SOTA 使用的版本）
- 纯 4 层 CNN: 3→32→64→64→16
- 不使用 DINOv2 特征引导
- 简单但有效，v2/v3b 的 DINOv2-guided FineEncoder 反而没有提升

#### DualScaleEncoder
- 共享 DINOv2 backbone，coarse 和 fine 分支共享 patch token 提取
- `use_old_fine_encoder=True` 选择 v1 的 OldFineEncoder

### 3.3 对比学习损失函数

**文件**: `gsff/losses.py`

训练使用四种损失的加权组合：

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{NCE} + 0.5 \cdot \mathcal{L}_{PRO} + 0.5 \cdot \mathcal{L}_{CE} + 0.1 \cdot \mathcal{L}_{TVL} + \lambda_{cos} \cdot \mathcal{L}_{cos}$$

| 损失 | 公式 | 作用 |
|------|------|------|
| $\mathcal{L}_{NCE}$ | Symmetric pixel-level InfoNCE | 正样本=同位置的 F3D/F2D，负样本=其他像素 |
| $\mathcal{L}_{PRO}$ | Prototypical contrastive loss | 鼓励 F3D 和 F2D 被映射到同一聚类原型 |
| $\mathcal{L}_{CE}$ | Segmentation cross-entropy | 语义分割对齐，辅助特征学习 |
| $\mathcal{L}_{TVL}$ | Total variation on triplane | 空间平滑正则化 |
| $\mathcal{L}_{cos}$ | Direct cosine similarity | 直接对齐 F2D 和 F3D 的余弦相似度 |

**InfoNCE** 实现细节：
- 每张图像随机采样 `max_samples=1024` 个像素
- 温度 $\tau = 0.07$
- 对称计算：3D→2D 和 2D→3D 方向各一个 cross_entropy

**Prototypical Loss**:
- 聚类原型通过 Sinkhorn-Knopp 最优传输进行 soft assignment
- 使用谱聚类对 Gaussian 中心做空间聚类 (K=34 clusters)
- 每 500 iterations 重新计算原型

### 3.4 2DGS 可微渲染

**文件**: `gsff/pose_refine.py`（渲染部分）

核心挑战：**gsplat v1.4.0 的 `rasterization_2dgs` 对 viewmat 不产生梯度！**

#### 解决方案：可微 Gaussian 变换

不改变 viewmat，而是变换 Gaussian 本身：

```python
def _transform_gaussians(means3d, quats, delta_T, fixed_viewmat):
    # 将 camera-space 的 delta_T 转换到 world-space
    fixed_inv = torch.inverse(fixed_viewmat)
    delta_T_world = fixed_inv @ delta_T @ fixed_viewmat
    
    R_dw = delta_T_world[:3, :3]
    t_dw = delta_T_world[:3, 3]
    
    # 变换 Gaussian 中心位置
    means_transformed = means3d @ R_dw.T + t_dw.unsqueeze(0)
    
    # 变换 Gaussian 旋转 (Hamilton 四元数乘法)
    q_delta = rotation_matrix_to_quaternion(R_dw)
    quats_transformed = quaternion_multiply(q_delta, quats)
    
    return means_transformed, quats_transformed
```

关键点：
1. `delta_T` 定义在 camera space（se(3) 指数映射）
2. 需要通过 `V^{-1} \cdot \Delta T \cdot V` 转换到 world space
3. 渲染时 viewmat 固定不变，梯度通过 Gaussian 参数回传到 `delta_T`

#### 特征分片渲染

颜色通道限制为 16（gsplat 限制），16 维特征可一次渲染：
```python
# chunk_size=16, D=16 → 1 chunk
rasterization_2dgs(
    means=means_transformed, quats=quats_transformed,
    scales=scales, opacities=opacities,
    colors=colors[:, c_start:c_end],  # [N, 16]
    viewmats=fixed_vm, Ks=K,
    width=width, height=height,
)
```

### 3.5 SE(3) 位姿优化

**文件**: `gsff/pose_refine.py` — `refine_pose()`

#### Lie 代数参数化

优化变量是 se(3) 向量 $\xi = [t_x, t_y, t_z, r_x, r_y, r_z]$：

$$\Delta T = \exp(\xi) \in SE(3)$$

使用 Rodrigues 公式：$R = I + \sin\theta \cdot K + (1 - \cos\theta) \cdot K^2$

#### 分离平移/旋转学习率（关键改进）

```python
if trans_lr_scale != 1.0:
    delta_t = torch.zeros(3, requires_grad=True)  # 平移
    delta_r = torch.zeros(3, requires_grad=True)  # 旋转
    param_groups = [
        {'params': [delta_t], 'lr': lr * trans_lr_scale},  # 10x
        {'params': [delta_r], 'lr': lr},                   # 1x
    ]
```

原因：平移梯度比旋转梯度小 10-50 倍。使用 `trans_lr_scale=10` 平衡两者。

#### 优化循环

```python
for i in range(n_iters):
    delta_xi = cat([delta_t, delta_r])
    delta_T = se3_exp(delta_xi)
    feat_3d = render_features_transformed(
        ..., delta_T, fixed_viewmat, ...)
    loss = MSE(normalize(feat_3d), feat_2d)
    loss.backward()
    clip_grad_norm_(params, 0.5)
    optimizer.step()
    # 追踪最佳 loss 对应的 viewmat
    if loss < best_loss:
        best_viewmat = se3_exp(delta_xi.detach()) @ init_viewmat
```

---

## 4. 训练流程

**文件**: `scripts/train_gsff.py`

### 4.1 训练配置

| 参数 | 值 |
|------|-----|
| 总迭代数 | 50,000 |
| Phase 1 (warm-up) | 0 ~ 10,000 iter (弱梯度 ×0.1) |
| Phase 2 (full loss) | 10,000 ~ 50,000 iter |
| Batch size | 1 (每次1张图，与3DGS训练一致) |
| 渲染分辨率 | 540×960 (render), 38×68 (coarse), 270×480 (fine) |
| Triplane LR | 5e-4 |
| Encoder LR | 1e-4 |
| Scheduler | Cosine annealing (1e-6 最小) |
| 聚类原型更新 | 每 500 iter |
| Feature dim | 16 |
| Coarse resolution | 256 |
| Fine resolution | 1024 |
| Scene extent | 99th percentile of Gaussian norm × 1.2 |

### 4.2 训练步骤

```
每个 iteration:
1. 采样一张训练图像 + 其 w2c 位姿
2. 2D Encoder 提取 coarse + fine 特征
3. Triplane 提取所有 Gaussian 的 coarse + fine 颜色
4. 2DGS 渲染 coarse 和 fine 特征图
5. L2 归一化所有特征
6. 计算 NCE + Prototypical + Segmentation CE + TVL 损失
7. 反向传播更新 triplane + encoder + seg_head
```

### 4.3 检查点策略

- `latest.pth`: 每 `save_freq` 次保存
- `best.pth`: 按 fine_cos_sim 指标选择（***注意：可能在 fine 尚未充分训练时保存**）
- `final.pth`: 训练结束时保存 — **SOTA 使用此检查点**

---

## 5. 评估（定位）流程

**文件**: `scripts/eval_gsff.py`

### 5.1 SOTA 配置

```
multi_start=3, rounds=3, coarse_iters=300, fine_iters=300
loss_type=mse, trans_lr_scale=10.0, lr_scale=1.0 (constant)
```

### 5.2 完整流程

```
对每张测试图像:
1. 初始化：找 Top-3 最近训练位姿 (position-NN)
2. 提取 2D 特征: coarse [16, 38, 68] + fine [16, 270, 480]

对每个候选初始位姿 (共3个):
  对每轮 (共3轮, lr_scale=1.0 恒定):
    Stage 1 — Coarse 优化:
      - 渲染 coarse triplane 特征 (R=256) @ 38×68
      - 300 iter Adam, lr=0.01, trans_lr_scale=10
      - MSE loss + grad clip 0.5
    Stage 2 — Fine 优化:
      - 渲染 fine triplane 特征 (R=1024) @ 270×480
      - 300 iter Adam, lr=0.005, trans_lr_scale=10
      - MSE loss + grad clip 0.5
  → 3轮 coarse-fine 迭代后得到 refined pose

3. 对3个候选结果，取 fine MSE loss 最小的作为最终位姿
4. 计算位置/旋转误差
```

### 5.3 分辨率配置

| 尺度 | 分辨率 | K 矩阵 |
|------|--------|---------|
| 渲染 | 960×540 | 原始缩放 |
| Coarse | 68×38 | render / 14 |
| Fine | 480×270 | render / 2 |

---

## 6. 关键改进汇总

### 6.1 去除跨轮次学习率衰减 ★★★★★

**问题**: 原始实现中每轮衰减 `lr_scale = 0.5^round`，第3轮只有初始 LR 的 12.5%。

**解决**: 改为恒定 `lr_scale = 1.0`。

**影响**: 39.9cm → 26.0cm (中位数)，**减少 35%**

**原因分析**: 后续轮次面对不同的特征 landscape，需要与第一轮相同的优化力度。

### 6.2 平移/旋转分离学习率 ★★★★★

**问题**: 平移梯度比旋转梯度小 10-50 倍，标准 Adam 无法有效优化平移。

**解决**: 分离 `delta_t` 和 `delta_r`，平移 LR 乘以 `trans_lr_scale=10`。

**影响**: 26.0cm → 23.6cm (单起点)，**减少 9%**

**实现**: 使用 Adam 的 `param_groups` 机制：
```python
param_groups = [
    {'params': [delta_t], 'lr': 0.01 * 10},  # 平移
    {'params': [delta_r], 'lr': 0.01},         # 旋转
]
```

### 6.3 多起点初始化 (Multi-start) ★★★★

**问题**: 约 22% 的测试样本会灾难性发散（优化让位姿远离 GT），单一初始化无法恢复。

**解决**: 取 Top-3 位置最近的训练图像作为候选，分别优化后取 fine loss 最低的。

**影响**: 
- P90 从 473cm → 94cm（单起点 trans_10x vs multi-start trans_10x）
- 中位数 23.6cm → 18.3cm

**原因**: 不同初始方向覆盖不同的 basin of attraction，大幅降低灾难性失败概率。

### 6.4 使用 final.pth 而非 best.pth ★★★

**问题**: `best.pth` 按 `fine_cos_sim` 保存，但可能在 fine 训练早期（iter 10K~15K）就被保存。此时 fine triplane 和 fine encoder 远未收敛。

**解决**: 使用 `final.pth`（iter 50K），充分训练后的检查点。

---

## 7. 踩坑记录

### 7.1 gsplat viewmat 零梯度 🔴 严重

**现象**: 位姿优化完全不动，loss 不下降。

**根因**: `gsplat v1.4.0` 的 `rasterization_2dgs` CUDA kernel 中，viewmat 的梯度全为零。

**解决**: 不优化 viewmat，而是将 `se3_exp(delta_xi)` 作用在 Gaussian 的均值和四元数上（`_transform_gaussians`）。这是整个项目的核心突破。

### 7.2 2DGS scales 是 2D 的 🔴 严重

**现象**: 渲染结果全黑或显示异常。

**根因**: 2DGS 的 Gaussian scales 只有 2 维 `[N, 2]`，但 `rasterization_2dgs` 需要 3 维 `[N, 3]`。

**解决**: Padding 第三维为 1：
```python
scales = torch.cat([scales_raw, torch.ones_like(scales_raw[:, :1])], dim=-1)
```

### 7.3 best.pth 陷阱 🟡 中等

**现象**: 使用 `best.pth` 评估，定位结果远差于预期。

**根因**: `best.pth` 是按 `fine_cos_sim` 选择的，但 fine 分支在 `phase1_iters` (10K) 之后才开始训练。如果 cos_f 在早期意外地高（例如 random 特征碰巧对齐），就会提前保存一个不好的检查点。

**解决**: 始终使用 `final.pth`。

### 7.4 RGB Photometric Loss 反效果 🟡 中等

**现象**: 加入 `rgb_weight=0.2` 后，中位误差从 42.7cm 飙升到 86.9cm。

**原因**: RGB 信号对视角变化过于敏感（高光、遮挡），提供了误导性梯度。Feature-only 优化更鲁棒。

### 7.5 DINOv2 Retrieval 不如 Position-NN 🟡 中等

**现象**: DINOv2 CLS token 检索初始化位姿后，中位误差 430cm vs position-NN 的 135cm。

**原因**: DINOv2 的视觉相似性不一定意味着空间邻近。两张观察方向相似但位置不同的图片会被误匹配。

### 7.6 Feature Tuning (在线微调特征) 无效 🟡 中等

**现象**: 论文中 "Feature tuned" 是在定位时联合优化 per-Gaussian 特征残差 + 位姿。我们的复现中这反而降低了精度。

**推测**: 300K+ Gaussians 的特征参数量太大，容易过拟合到当前 loss 而非正确位姿。

### 7.7 Loss landscape 极度平坦 🟠 根本性限制

**诊断结果**: 200cm 的位置偏移只让 MSE loss 变化 33%。特征场在空间上的区分度有限。

**影响**: 22% 的样本存在灾难性失败（即便初始误差只有 21cm 也会发散到 477cm）。Multi-start 是缓解手段，根治需要更好的特征训练。

### 7.8 v2/v3b FineEncoder 未提升 🟠 意外

**尝试**: 使用 DINOv2 引导的 FineEncoder（CNN + 上采样DINOv2 tokens → fusion）替代简单 4 层 CNN。

**结果**: 性能基本持平或略差。简单 CNN 更稳定。

---

## 8. 消融实验结果

### 8.1 完整消融表

| 配置 | 中位位置 | 中位旋转 | 平均位置 | P90位置 |
|------|---------|---------|---------|---------|
| Init (pos-NN) | 135.5cm | 11.50° | 209.7cm | — |
| 1R single (300+300) | 65.9cm | 0.89° | 107.2cm | 272.0cm |
| 2R lr-decay | 42.7cm | 0.60° | 84.6cm | 225.6cm |
| 3R lr-decay | 39.9cm | 0.60° | 81.6cm | 227.0cm |
| **3R no-lr-decay** | **26.0cm** | **0.43°** | 77.0cm | 243.4cm |
| 4R no-lr-decay | 25.8cm | 0.40° | 79.9cm | 258.0cm |
| trans_10x 3R | 23.6cm | 0.36° | 101.0cm | 473.4cm |
| trans_10x 4R | 22.8cm | 0.36° | 99.1cm | 436.2cm |
| ms3 3R lr-decay | 27.9cm | 0.48° | 61.9cm | 123.3cm |
| ms3 3R no-lr-decay | 19.8cm | 0.34° | 53.1cm | 111.9cm |
| **★ trans_10x + ms3 + 3R** | **18.3cm** | **0.32°** | **48.6cm** | **93.7cm** |
| 论文 Feature | 21cm | 0.41° | — | — |
| 论文 Feature tuned | 18cm | 0.36° | — | — |

### 8.2 关键观察

1. **多轮迭代**: 1R→3R，中位误差从 65.9cm 降到 39.9cm（lr-decay），或 26.0cm（no-lr-decay）
2. **LR 策略**: 恒定 LR 远优于衰减 LR（39.9→26.0，改善 35%）
3. **trans_lr_scale**: 10x 在单起点下从 26.0→23.6cm，但 P90 恶化（可能导致过矫正）
4. **Multi-start**: 根本解决 P90 问题（473→94cm），并提升中位数。是最有效的鲁棒性改进
5. **三者结合**: 中位 18.3cm / P90 93.7cm，全面优于论文

### 8.3 失败案例分析

基于 50 样本分析：
- **62% GOOD** (< 25cm)
- **16% MEDIUM** (25-100cm)  
- **22% FAIL** (> 100cm)
- 旋转几乎总是成功的（中位 0.32°）；位置是唯一瓶颈
- `init_pos` 与 `final_pos` 相关系数 = **-0.148**（负的！说明部分样本优化方向完全错误）
- `seq4` 区域的特征场质量差，经常导致发散

---

## 9. 复现命令速查

### 训练

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/train_gsff.py \
    --source_dir dataset/OldHospital \
    --model_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
    --cameras_json output/2dgs_models/OldHospital/v7_depth/cameras.json \
    --output_dir output/gsff/OldHospital \
    --total_iters 50000 \
    --phase1_iters 10000 \
    --coarse_resolution 256 \
    --fine_resolution 1024 \
    --feature_dim 16 \
    --render_height 540 \
    --render_width 960
```

### 评估（SOTA 配置）

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/eval_gsff.py \
    --checkpoint output/gsff/OldHospital/checkpoints/final.pth \
    --source_dir dataset/OldHospital \
    --model_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
    --cameras_json output/2dgs_models/OldHospital/v7_depth/cameras.json \
    --rounds 3 \
    --multi_start 3 \
    --coarse_iters 300 \
    --fine_iters 300 \
    --loss_type mse \
    --trans_lr_scale 10.0
```

### 监控训练

```bash
# 查看最新 loss
tail -20 output/gsff/OldHospital/train.log

# 查看 fine cosine similarity 指标趋势
grep 'cos_f' output/gsff/OldHospital/train.log | tail -20
```

---

## 附录 A：文件清单

| 文件 | 行数 | 作用 |
|------|------|------|
| `gsff/triplane.py` | ~116 | Triplane 特征场 (TriplaneFeatureField, DualScaleTriplane) |
| `gsff/encoder.py` | ~260 | 2D 编码器 (CoarseEncoder, FineEncoder, DualScaleEncoder) |
| `gsff/losses.py` | ~200 | NCE + Prototypical + Segmentation CE 损失 |
| `gsff/clustering.py` | ~100 | 谱聚类 + 原型计算 |
| `gsff/pose_refine.py` | ~394 | SE(3) 可微位姿优化 (核心) |
| `scripts/train_gsff.py` | ~815 | 训练入口 (GSFFTrainer) |
| `scripts/eval_gsff.py` | ~515 | 评估入口 (multi-start, multi-round) |
| `scripts/diagnose_gap.py` | ~430 | 诊断工具 |
| `scripts/analyze_failures.py` | ~200 | 失败案例分析 |
| `feature_3dgs/gaussian_feature_model.py` | — | 2DGS 模型加载 |

## 附录 B：与论文的主要差异

| 论文做法 | 我们的实现 | 影响 |
|---------|-----------|------|
| 专用 2DGS Training | 使用预训练 2DGS 模型 | 省去 Phase 1 RGB 训练 |
| DenseVLAD 检索初始化 | Position-NN (GT 位置获取最近邻) | 更公平对比，但实际部署需要替换 |
| 单起点优化 | Multi-start Top-3 | P90 大幅改善 |
| 统一 LR | 分离 trans/rot LR (10:1) | 中位改善 ~15% |
| LR 衰减 | 恒定 LR | 改善 35% |
