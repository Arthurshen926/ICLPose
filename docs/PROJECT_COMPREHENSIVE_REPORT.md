# ICLPose 项目综合文档报告

> **自动生成时间**: 2025  
> **Git 分支**: `v2-iterative-routing` (HEAD: `9e9a325`)  
> **Python 文件数**: 184  
> **核心框架**: PyTorch ≥2.0, gsplat v1.4+, DINOv2, Stable Diffusion (ODISE)

---

## 目录

1. [项目概览](#1-项目概览)
2. [目录结构](#2-目录结构)
3. [环境与依赖](#3-环境与依赖)
4. [模型架构](#4-模型架构)
5. [特征提取管线](#5-特征提取管线)
6. [特征压缩](#6-特征压缩)
7. [3DGS 特征嵌入与几何重建](#7-3dgs-特征嵌入与几何重建)
8. [数据管线](#8-数据管线)
9. [训练流程](#9-训练流程)
10. [损失函数](#10-损失函数)
11. [核心模块详解](#11-核心模块详解)
12. [工具与可视化](#12-工具与可视化)
13. [脚本索引](#13-脚本索引)
14. [SplatLoc 集成模块](#14-splatloc-集成模块)
15. [Git 历史与演进](#15-git-历史与演进)

---

## 1. 项目概览

**ICLPose** 是一个基于 **3D Gaussian Splatting (3DGS/2DGS)** 场景表示的 **6-DOF 相机位姿估计** 框架。核心思想是利用预训练视觉基础模型 (Stable Diffusion + DINOv2) 提取多尺度语义特征，将其嵌入到 3DGS 场景中，通过渲染-比较的方式建立隐式 2D-3D 对应关系，最终通过 RAFT 风格的迭代光流细化和几何求解器恢复精确位姿。

### 核心流水线

```
输入图像 → [SD+DINO多尺度特征提取] → [特征压缩(AutoEncoder)]
                                         ↓
3DGS场景 + 初始位姿 → [多尺度特征渲染] → [Coarse→Mid→Fine 层级关联]
                                         ↓
                                    [ConvGRU迭代细化] → [Image Jacobian几何求解] → δξ∈se(3)
                                         ↓
                                    [外循环迭代: 重新渲染 → 再次细化] → 最终位姿 T∈SE(3)
```

### 三种位姿估计架构

| 模型 | 文件 | 特点 |
|------|------|------|
| **MSFlowPoseNet** | `ic_models/ms_flow_pose_net.py` | **主架构**: Multi-Scale Flow + RAFT-style 迭代 + Image Jacobian 几何求解 |
| **ICPoseNet** | `ic_models/ic_pose_net.py` | 早期架构: Query-based 特征匹配 + Transformer |
| **ICPoseNetV3** | `ic_models/ic_pose_net_v3.py` | Render-and-Compare: 动态特征选择 + ConvGRU + 双头 |

---

## 2. 目录结构

```
ICLPose/
├── configs/                    # 实验配置 YAML (exp020-exp032)
├── data/                       # 数据集加载器
│   ├── dataset.py              # CorrespondenceDataset (RGB+3DGS PLY+2D-3D对应)
│   ├── dataset_v3.py           # PoseDatasetV3 (多尺度特征+位姿扰动)
│   ├── dataset_v4.py           # PoseDatasetV4 (深度+NetVLAD初始化)
│   └── dataset_sim.py          # SimPoseDataset (仿真数据+混合训练)
├── ic_models/                  # 位姿估计网络
│   ├── ms_flow_pose_net.py     # ★ MSFlowPoseNet (主模型, 676行)
│   ├── corr_pose_net.py        # CorrPoseNet (单尺度前身, 867行)
│   ├── ic_pose_net.py          # ICPoseNet (早期, 296行)
│   ├── ic_pose_net_iterative.py # ICPoseNetIterative (427行)
│   ├── ic_pose_net_v2.py       # ICPoseNetV2 (479行)
│   ├── ic_pose_net_v3.py       # ICPoseNetV3 (Render&Compare, 403行)
│   └── positional_encoding.py  # NeRF/Fourier PE (234行)
├── modules/                    # 可复用模块 (17个文件)
│   ├── geometry_solver.py      # Image Jacobian + 加权最小二乘
│   ├── lie_algebra.py          # SE(3)/SO(3) 指数/对数映射
│   ├── multiscale_renderer.py  # 多尺度 gsplat 渲染
│   ├── conv_gru.py             # ConvGRU 空间循环单元
│   ├── dual_head.py            # 双头: Flow → FlowToPose 几何 + PoseHead 回归
│   ├── dynamic_feature_selector.py # 尺度门控 + 残差计算 + Softmax 竞争
│   ├── featuremetric.py        # 纯几何 Gauss-Newton/LM 直接对齐
│   ├── flow_to_pose.py         # 光流→SE(3): Image Jacobian WLS
│   ├── fusion_module.py        # 自注意力+交叉注意力融合
│   ├── fusion_module_c2f.py    # Coarse-to-Fine 融合 + 3D PE
│   ├── overlap_detection.py    # 视锥重叠检测 + 位姿预测
│   ├── pose_aware_upsampler.py # 位姿感知上采样 (768d→64d)
│   ├── pose_regressor.py       # 6D旋转/四元数 位姿回归
│   ├── pose_regressor_marepo.py # MaRepo风格: SVD旋转 + 12层自注意力
│   └── transformer.py          # 多头注意力 (APE+RPE)
├── losses/                     # 损失函数 (4个文件)
│   ├── sequence_loss.py        # RAFT-style γ^(K-k) 位姿+光流序列损失
│   ├── pose_loss.py            # PoseLoss (测地线/余弦旋转 + L1/L2平移)
│   ├── pose_loss_c2f.py        # C2F 多阶段损失 + DynTanh 软裁剪
│   └── reprojection_loss.py    # 3D→2D 重投影损失
├── feature_extraction/         # 特征提取器
│   ├── extractor_dino.py       # DINOv2 ViT-B/14 特征提取
│   ├── extractor_sd.py         # ODISE SD UNet 特征提取
│   ├── fused_feature_extractor.py # SD+DINO+AggregationNetwork 融合
│   ├── multiscale_extractor.py # ★ 多尺度独立提取 (fine_sd/fine_dino/mid/coarse/cls)
│   ├── projection_network.py   # ODISE AggregationNetwork (ResNet Bottleneck)
│   └── resnet.py               # ResNet BottleneckBlock (Detectron2)
├── feature_compression/        # 特征压缩
│   ├── autoencoder.py          # AutoencoderFlexible (L2+MinMax归一化)
│   ├── compressor.py           # FeatureCompressor 包装器
│   └── eval_compression.py     # 压缩质量评估
├── feature_3dgs/               # ★ 3DGS 特征嵌入 & 2DGS 几何重建
│   ├── train_2dgs_geometry.py  # ★ 2DGS RGB 重建训练 (2700行, 含WildGaussians)
│   ├── train_multiscale_embedding_v2.py # 多尺度特征嵌入训练 (加速版)
│   ├── train_raw_embedding.py  # 原始维度按尺度独立嵌入训练
│   ├── gaussian_feature_model.py # 冻结几何 + 可训练特征的 3DGS
│   ├── feature_renderer.py     # gsplat v1.4+ 特征渲染器
│   ├── multiscale_gaussian_model.py # 224d 多尺度拆分模型
│   ├── raw_gaussian_model.py   # 原始维度 per-scale 模型
│   ├── multiscale_dataset.py   # 压缩特征数据集
│   ├── raw_multiscale_dataset.py # 原始特征数据集
│   ├── feature_3dgs_provider.py # 深度反投影 + 特征采样 Provider
│   ├── feature_dataset.py      # 单尺度特征嵌入数据集
│   ├── eval_feature_pca.py     # PCA 可视化评估
│   ├── eval_rendering.py       # 渲染质量评估 (L1/CosSim/PSNR)
│   └── verify_pipeline.py      # 管线验证
├── splatloc_modules/           # SplatLoc 集成
│   ├── models/
│   │   ├── decoders.py         # FeatureDecoder (HashGrid MLP)
│   │   └── encoding.py         # tinycudann Grid/Hash/SH/频率编码
│   └── gaussian_splatting/     # 3DGS 基础设施 (680行 GaussianModel)
├── utils/                      # 工具函数
│   ├── model_factory.py        # create_model() 工厂
│   ├── loss_factory.py         # create_loss_function() + CombinedLoss
│   ├── training_utils.py       # 位姿工具 + GradientClipper + EMA + MetricLogger
│   └── visualization.py        # 注意力图/对应关系/特征相似度可视化
├── scripts/                    # 脚本 (82 Python + 58 Shell)
│   ├── train_ms_flow.py        # ★ MSFlowPoseNet 主训练脚本 (887行)
│   ├── train_corr_pose.py      # CorrPoseNet 训练
│   ├── train_multiscale_ae.py  # AutoEncoder 训练
│   ├── extract_features_v2.py  # 特征提取脚本
│   └── ...                     # 评估/诊断/可视化/benchmark 脚本
├── train_v3.py                 # ICPoseNetV3 训练脚本
├── train.py                    # 通用训练入口
└── train_distributed.sh        # 多GPU分布式训练
```

---

## 3. 环境与依赖

### Conda 环境: `geo-aware`
- **Python**: 3.9
- **CUDA**: 11.6
- **路径**: `/home/yons/.conda/envs/geo-aware`

### 核心依赖 (requirements.txt)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
PyYAML>=6.0
tqdm>=4.65.0
pillow>=9.5.0
opencv-python>=4.7.0
matplotlib>=3.7.0
tensorboard>=2.13.0
plyfile>=0.7.4
open3d>=0.17.0
trimesh>=3.21.0
scikit-learn>=1.2.0
diff-gaussian-rasterization>=0.0.0  # Custom package
simple-knn>=0.0.0                    # Custom package
munch>=2.5.0
einops>=0.6.1
```

### 额外关键依赖 (代码中直接引用)
- `gsplat` v1.4+ — 3DGS/2DGS 光栅化 (`rasterization_2dgs`, `spherical_harmonics`)
- `tinycudann` — HashGrid/Dense Grid 编码 (SplatLoc 模块)
- DINOv2 ViT-B/14 — `facebookresearch/dinov2` (本地缓存)
- ODISE SD UNet — Stable Diffusion 特征提取

---

## 4. 模型架构

### 4.1 MSFlowPoseNet (主模型)

**文件**: `ic_models/ms_flow_pose_net.py` (676行)

```
[Query多尺度特征]     [3DGS渲染多尺度特征]
  coarse(32d,7×10)      coarse(32d,7×10)
  mid(64d,15×20)        mid(64d,15×20)
  fine_sd(64d,35×46)    fine_sd(64d,35×46)
  fine_dino(64d,35×46)  fine_dino(64d,35×46)
        ↓                       ↓
    ScaleDecoder            ScaleDecoder
    (1×1 Conv→64d)          (1×1 Conv→64d)
        ↓                       ↓
  ┌─────────────────────────────────┐
  │ Stage 1: Coarse Flow (7×10)    │
  │   Global all-pairs correlation │
  │   → FlowRefinementHead        │
  │   → coarse_flow + confidence   │
  └─────────┬───────────────────────┘
            ↓ (warp guide)
  ┌─────────────────────────────────┐
  │ Stage 2: Mid Flow (15×20)      │
  │   Warp-guided local corr r=4   │
  │   → FlowRefinementHead        │
  │   → mid_flow + confidence      │
  └─────────┬───────────────────────┘
            ↓ (warp guide)
  ┌─────────────────────────────────┐
  │ Stage 3: Fine Flow (35×46)     │
  │   RAFT-style iterative:        │
  │   for k in range(fine_iters):  │
  │     local_corr → encoder →    │
  │     ConvGRU → Δflow + conf    │
  │   FineDualDecoder: SD + DINO   │
  └─────────┬───────────────────────┘
            ↓
  ┌─────────────────────────────────┐
  │ Geometry Solver (35×46)         │
  │   compute_image_jacobian(flow,  │
  │     depth, K) → Ju, Jv         │
  │   diff_pose_solve(J, flow,     │
  │     confidence) → δξ∈se(3)    │
  │   LM damping: λ=0.01          │
  └─────────┬───────────────────────┘
            ↓
        se3_exp(δξ) → ΔT ∈ SE(3)
        T_{k+1} = ΔT · T_k
```

#### 关键组件

**ScaleDecoder**: 每个尺度的 1×1 Conv 投影到统一 64d 维度

**FineDualDecoder**: Fine 层拆分 SD + DINO 双流
- SD通道: `nn.Conv2d(64, 64, 1)` → corr → flow
- DINO通道: `nn.Conv2d(64, 64, 1)` → corr → flow
- 融合: `cat(sd_corr, dino_corr)` → shared encoder → ConvGRU

**FlowRefinementHead**: 
```python
correlation → encoder(Conv2d stack) → ConvGRU hidden → flow_head(Δflow + confidence)
```

**Image Jacobian 几何求解**:
```python
# modules/geometry_solver.py
Ju = -fx/Z, Jv = -fy/Z  # 平移部分
# 旋转部分: 交叉积形式的 Jacobian
J = [Ju; Jv]  # [N, 2, 6]
# 加权最小二乘: (J^T W J + λI)^{-1} J^T W r
```

**外循环迭代** (Outer Iterations):
```python
for outer in range(outer_iters):
    # 1. 用当前位姿渲染多尺度特征
    rendered = multiscale_renderer.render(current_pose)
    # 2. 前向: query vs rendered → flow → δξ
    delta_pose = model(query_feats, rendered, depth, K)
    # 3. 更新位姿: T = exp(δξ) · T
    current_pose = se3_exp(delta) @ current_pose
```

### 4.2 CorrPoseNet (单尺度前身)

**文件**: `ic_models/corr_pose_net.py` (867行)

单尺度版本，使用 local_correlation + soft-argmax + learnable damping。包含完整的 RAFT-style ConvGRU 循环。

### 4.3 ICPoseNetV3 (Render-and-Compare)

**文件**: `ic_models/ic_pose_net_v3.py` (403行)

- **DynamicFeatureSelector**: 多尺度门控 + softmax 竞争
- **ConvGRU**: 迭代状态更新
- **DualHead**: Flow → FlowToPose (几何) 或 PoseHead (回归) 双路径

---

## 5. 特征提取管线

### 5.1 多尺度特征提取器

**文件**: `feature_extraction/multiscale_extractor.py` (300行)

提取 5 种独立特征（不做融合）:

| 特征名 | 来源 | 维度 | 分辨率 | 说明 |
|--------|------|------|--------|------|
| `fine_sd` | SD UNet s3 | 640d | 35×46 (v1) / 32×40 (v2) | Stable Diffusion 浅层 |
| `fine_dino` | DINOv2 patch tokens | 768d | 35×46 | ViT-B/14 局部特征 |
| `mid` | SD UNet s4 | 1280d | 15×20 (v1) / 16×20 (v2) | SD中层语义 |
| `coarse` | SD UNet s5 | 1280d | 7×10 (v1) / 8×10 (v2) | SD深层全局 |
| `cls_token` | DINOv2 CLS | 768d | 1×1 | 全局描述子 (用于place recognition) |

**v2 改进**: SD 特征保留 UNet 内部零填充后的原生分辨率，形成干净的 2× 层级: coarse 8×10 → mid 16×20 → fine_sd 32×40

### 5.2 SD 特征提取 (ODISE)

**文件**: `feature_extraction/extractor_sd.py`

```python
class StableDiffusionSeg:
    # 基于 ODISE 的 SD UNet 特征提取
    # 输入: RGB图像 → SD encode → 加噪 → UNet单步 denoise → 提取中间层特征
    # 输出: s3(640d), s4(1280d), s5(1280d) at different resolutions
```

**关键**: SD 输入会先做零填充到 64 的倍数，提取后需裁剪回原始尺寸。

### 5.3 DINO 特征提取

**文件**: `feature_extraction/extractor_dino.py`

```python
class ViTExtractor:
    # DINOv2 ViT-B/14 特征提取
    # 输入: RGB → resize to patch-aligned → ViT forward
    # 输出: patch_tokens [N_patches, 768], cls_token [768]
    # 支持 stride 和 position encoding 插值
```

### 5.4 融合特征提取器

**文件**: `feature_extraction/fused_feature_extractor.py`

```python
class FusedFeatureExtractor:
    # 加载 SD + DINO + AggregationNetwork
    # 将 SD 多层特征与 DINO 特征通过 learnable mixing weights 融合为 768d
    # 用于旧版单尺度模型
```

### 5.5 AggregationNetwork

**文件**: `feature_extraction/projection_network.py`

ODISE 风格的多层特征聚合网络:
- ResNet BottleneckBlock 处理各层特征
- learnable softmax mixing weights 混合各层
- 640+1280+1280+768 → 768d 投影
- contrastive temperature parameter

---

## 6. 特征压缩

### 6.1 AutoencoderFlexible

**文件**: `feature_compression/autoencoder.py`

```python
FEATURE_CONFIGS = {
    'v2_fine_sd':   {'input_dim': 640,  'bottleneck_dim': 64},
    'v2_fine_dino': {'input_dim': 768,  'bottleneck_dim': 64},
    'v2_mid':       {'input_dim': 1280, 'bottleneck_dim': 64},
    'v2_coarse':    {'input_dim': 1280, 'bottleneck_dim': 32},
    # Legacy configs:
    'dino':  {'input_dim': 768,  'bottleneck_dim': 64},
    'fused': {'input_dim': 768,  'bottleneck_dim': 256},
    'sd_s3': {'input_dim': 640,  'bottleneck_dim': 64},
    'sd_s4': {'input_dim': 1280, 'bottleneck_dim': 64},
    'sd_s5': {'input_dim': 1280, 'bottleneck_dim': 32},
}
```

**编码流程**:
1. L2 归一化 (per-pixel)
2. Linear encoder → bottleneck
3. Min-Max 归一化到 [0, 1] (使用 calibrate() 预计算的统计量)

**压缩比例**:
- fine_sd: 640d → 64d (10×)
- fine_dino: 768d → 64d (12×)
- mid: 1280d → 64d (20×)
- coarse: 1280d → 32d (40×)

---

## 7. 3DGS 特征嵌入与几何重建

### 7.1 2DGS 几何重建 (★ 核心文件)

**文件**: `feature_3dgs/train_2dgs_geometry.py` (2700行, 当前修改中)

独立的 2DGS RGB 几何重建训练脚本，核心特性:

- **2DGS 渲染**: gsplat `rasterization_2dgs` (surfel ray-intersection)
- **损失组合**: L1 + SSIM + distortion + normal consistency + Pearson depth + scale reg + opacity entropy
- **Adaptive densification**: clone + split + prune + floater suppression + post-densification pruning
- **批量渲染**: `render_2dgs_batch()` 单次 CUDA kernel 多视角渲染
- **三种 Appearance 模式**:
  1. **AppearanceNetwork**: 全局 per-image 仿射颜色校正 (scale ∈ [0.8, 1.2], bias ∈ [-0.05, 0.05])
  2. **SpatialAppearanceNetwork**: CNN decoder 生成逐像素 scale/bias 图 (处理局部阴影/高光)
  3. **WildGaussiansAppearance** (NeurIPS 2024): per-Gaussian × per-Image MLP + Fourier 位置编码 + DINO 不确定性
- **DINO 不确定性掩码**: DinoUncertaintyPredictor 用 DINOv2 余弦相似度检测动态物体

**GaussianModel2DGS 参数**:
```python
_xyz:          [N, 3]  位置
_features_dc:  [N, 1, 3]  SH DC 系数
_features_rest:[N, K, 3]  SH 高阶系数
_scaling:      [N, 2]  2D 尺度 (log space)
_rotation:     [N, 4]  四元数旋转
_opacity:      [N, 1]  不透明度 (sigmoid space)
```

### 7.2 GaussianFeatureModel

**文件**: `feature_3dgs/gaussian_feature_model.py` (249行)

冻结 3DGS 几何 + 可训练 per-Gaussian 特征嵌入:
```python
class GaussianFeatureModel:
    _xyz, _scaling, _rotation, _opacity  # 冻结 (from PLY)
    _loc_feature: nn.Parameter  # [N, feature_dim] 可训练
```

### 7.3 FeatureRenderer

**文件**: `feature_3dgs/feature_renderer.py` (391行)

基于 gsplat v1.4+ 的特征图渲染:
- 自动检测 3DGS (3D scales) vs 2DGS (2D scales)
- Channel chunking: 每次渲染 ≤32 通道 (CUDA shared memory 限制)
- 批量渲染支持

### 7.4 MultiScaleGaussianModel

**文件**: `feature_3dgs/multiscale_gaussian_model.py`

```python
# 224d 总特征向量拆分:
[fine_sd(64) | fine_dino(64) | mid(64) | coarse(32)]
# 渲染后按偏移量拆分出 3 个尺度的特征图
```

### 7.5 Feature3DGSProvider

**文件**: `feature_3dgs/feature_3dgs_provider.py` (428行)

位姿估计训练的数据提供器:
```python
render_and_backproject(c2w, fx, fy, cx, cy, H, W, num_samples):
    # 1. 渲染深度图 → 有效深度像素筛选
    # 2. 深度反投影 → 世界坐标系 3D 点
    # 3. 渲染特征图 → 在像素位置采样 3D 特征
    # 返回: points_3d, pixel_coords, pcd_feats, depth_map, feature_map
```

---

## 8. 数据管线

### 8.1 PoseDatasetV4 (主训练数据集)

**文件**: `data/dataset_v4.py` (274行)

```python
class PoseDatasetV4(Dataset):
    # 加载多尺度压缩特征 + GT w2c 位姿 + 深度图
    # 训练: GT + 随机 se(3) 扰动 (noise curriculum)
    # 验证: NetVLAD 检索的初始位姿
    # 自动检测 v1/v2 特征格式
    # 返回: query_feats (dict by scale), pose_gt, initial_pose, depth
```

### 8.2 PoseDatasetV3 (V3模型数据集)

**文件**: `data/dataset_v3.py` (258行)

为 ICPoseNetV3 设计，支持:
- 多尺度原始特征 (未压缩)
- NetVLAD 初始位姿加载
- 深度图 (用于 flow loss)

### 8.3 CorrespondenceDataset (早期)

**文件**: `data/dataset.py` (816行)

完整的 2D-3D 对应数据集:
- RGB 图像 + 3DGS PLY 加载
- 视锥剔除 (frustum culling)
- 2D-3D 配对生成 (含负样本)
- 融合特征加载
- 噪声增强

### 8.4 SimPoseDataset (仿真训练)

**文件**: `data/dataset_sim.py` (203行)

3DGS 渲染的仿真数据:
- **MixedPoseDataset**: 真实数据 + 仿真数据按比例混合 (默认 real_ratio=0.3)
- 每 epoch 重新采样

---

## 9. 训练流程

### 9.1 MSFlowPoseNet 训练 (主流程)

**文件**: `scripts/train_ms_flow.py` (887行)

#### 两阶段训练

**Phase 1** (flow-only warmup, 默认 3 epochs):
- 只训练光流预测，不传播位姿梯度
- 目的: 让 ConvGRU 学会基本的光流估计

**Phase 2** (flow + pose, 渐进式):
- 位姿损失权重从 0 线性增长到 `pose_weight`
- 光流和位姿联合优化

#### 噪声课程 (Noise Curriculum)

```python
noise_curriculum:
    warmup_epochs: 40
    noise_rot_start: 30.0°  # 初始大噪声
    noise_rot_end: 5.0°     # 最终小噪声
    noise_trans_start: 1.5m
    noise_trans_end: 0.3m
```

#### 序列损失 (RAFT-style)

```python
# losses/sequence_loss.py
for k in range(K):  # K = fine_iters (4~8)
    loss_k = γ^(K-1-k) * (pose_loss_k + flow_loss_k)
    # γ=0.85, 越靠后的迭代权重越大
```

#### 训练配置 (exp032)

```yaml
# configs/exp032_cosine_fiters8.yaml
model:
  fine_iters: 8              # RAFT ConvGRU 迭代次数
  corr_radius: 4             # 局部关联半径
training:
  outer_iters: 3             # 训练时外循环次数
  val_outer_iters: 5         # 验证时外循环次数
  lr: 5e-5
  epochs: 60
  rot_loss_type: cosine      # 旋转损失类型
  trans_weight: 10.0         # 平移损失权重
  phase1_epochs: 3           # Flow-only warmup
  noise_curriculum:
    warmup_epochs: 40
```

### 9.2 ICPoseNetV3 训练

**文件**: `train_v3.py`

迭代 Render-and-Compare 训练:
- `forward_with_prerendered()`: 预渲染特征提高效率
- DynamicFeatureSelector 学习尺度选择

### 9.3 2DGS 几何训练

**文件**: `feature_3dgs/train_2dgs_geometry.py`

```bash
CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_geometry \
    --source_dir dataset/OldHospital \
    --model_dir output/2dgs_models/OldHospital/v3 \
    --images processed \
    --iterations 30000 \
    --longest_edge 1280 \
    --batch_size 4 \
    --use_appearance --wildgaussians \
    --lambda_dist 0.01 --lambda_normal 0.05 \
    --mono_depth_dir dataset/OldHospital/mono_depth
```

---

## 10. 损失函数

### 10.1 SequenceLoss (RAFT-style)

**文件**: `losses/sequence_loss.py` (360行)

```python
class SequenceLoss:
    # 总损失 = Σ_k γ^(K-1-k) · (w_pose·pose_loss_k + w_flow·flow_loss_k)
    # γ = 0.85 (指数递增权重, 后面迭代更重要)
    # pose_loss: rotation_geodesic + trans_weight * L1_translation
    # flow_loss: confidence-weighted L1 + masked flow L1
```

### 10.2 PoseLoss

**文件**: `losses/pose_loss.py` (663行)

```python
class PoseLoss:
    # 旋转: geodesic (acos) / L2 / cosine / quaternion
    # 平移: L1 / L2 / smooth_L1
    # 强制 fp32 避免 AMP half-precision 数值问题

class PoseLossKendall:
    # Kendall自动加权: exp(-log_var) * loss + log_var
    # 可学习的 log_var 自动平衡旋转/平移
```

### 10.3 PoseLossC2F

**文件**: `losses/pose_loss_c2f.py` (410行)

```python
class PoseLossC2F:
    # 多阶段权重: [w_coarse, w_fine1, w_fine2, ...]
    # 旋转 warmup: L1 → geodesic 渐进切换 (避免 acos 初期不稳定)
    # DynTanh 软裁剪: 限制 outlier 梯度
```

### 10.4 ReprojectionLoss

**文件**: `losses/reprojection_loss.py` (281行)

```python
# 3D→2D 重投影: P_cam = T · P_world → pixel = K · P_cam / Z
# L1/L2 + 有效深度掩码 + 可选置信度加权
```

### 10.5 2DGS 几何损失

```python
# train_2dgs_geometry.py 中定义
total = (1-λ_dssim)·L1 + λ_dssim·DSSIM          # RGB
      + λ_normal · normal_consistency              # 法线一致性
      + λ_dist   · distortion                      # 2DGS 正则化
      + λ_depth  · pearson_depth_loss              # 单目深度 Pearson 相关
      + λ_scale  · (max_log_scale - threshold)²    # 尺度正则 (抑制 floater)
      + λ_oe     · opacity_entropy                 # 不透明度熵 (鼓励二值化)
```

---

## 11. 核心模块详解

### 11.1 Lie 代数

**文件**: `modules/lie_algebra.py` (343行)

```python
hat(v)           # R³ → so(3)  反对称矩阵
so3_exp(omega)   # so(3) → SO(3) Rodrigues 公式
se3_exp(xi)      # se(3) → SE(3)  ξ=[t,ω] → T∈SE(3)
se3_log(T)       # SE(3) → se(3)
pose_compose(T1, T2)  # T1 · T2
pose_inverse(T)       # T^(-1)
compute_gt_flow(T_delta, depth, K)  # ΔT + depth → 2D flow field
```

### 11.2 几何求解器

**文件**: `modules/geometry_solver.py` (144行)

```python
compute_image_jacobian(flow, depth, K):
    # 每像素的 Image Jacobian: J ∈ R^{2×6}
    # Ju = -fx/Z · I₂ + ...  (平移+旋转部分)
    # 返回: Ju, Jv, valid_mask

diff_pose_solve(J, residual, confidence):
    # 加权最小二乘 + LM damping
    # (J^T W J + λI)^{-1} J^T W r → δξ ∈ se(3)
```

### 11.3 ConvGRU

**文件**: `modules/conv_gru.py` (152行)

```python
class ConvGRUCell:
    # 3×3 spatial gated recurrent unit
    # z = σ(W_z * [h, x])            # update gate
    # r = σ(W_r * [h, x])            # reset gate  
    # h̃ = tanh(W * [r⊙h, x])        # candidate
    # h = (1-z)⊙h + z⊙h̃             # new hidden

class ConvGRUBlock:
    # input_encoder (conv stack) → ConvGRUCell
```

### 11.4 Flow → Pose 转换

**文件**: `modules/flow_to_pose.py` (216行)

```python
class DifferentiableFlowToPose:
    # 从 2D 光流 + 深度 + 内参 → SE(3) 位姿增量
    # 1. 构建 Image Jacobian J(u,v,Z,K)
    # 2. 加权最小二乘求解: (J^T W J)^{-1} J^T W · flow
    # 3. se3_exp → SE(3)
```

### 11.5 Featuremetric 直接对齐

**文件**: `modules/featuremetric.py` (673行)

```python
class FeaturemetricAligner:
    # 纯几何方法 (无需训练)
    # Gauss-Newton/LM 迭代: 最小化特征重投影误差
    # align_fast(): Cholesky 分解, 无 .item() GPU 同步, 缓存 Jacobian
    # 自适应阻尼 + 早停
```

### 11.6 动态特征选择器

**文件**: `modules/dynamic_feature_selector.py` (255行)

```python
class DynamicFeatureSelector:
    # ScaleGate: GAP → MLP → sigmoid (per-scale 重要性权重)
    # ResidualComputer: subtract/concat/correlation 模式
    # 多尺度门控融合 + softmax 尺度竞争
```

### 11.7 多尺度渲染器

**文件**: `modules/multiscale_renderer.py` (336行)

```python
class MultiscaleRenderer:
    # 管理多个 GaussianFeatureModel 实例 (per-scale)
    # 或单个 MultiScaleGaussianModel (224d 统一渲染)
    # 批量 gsplat 渲染 → 按尺度拆分
    # v1/v2 分辨率配置
```

### 11.8 Overlap Detection

**文件**: `modules/overlap_detection.py` (524行)

```python
class OverlapDetectionModule:
    # OverlapEstimator: 特征相似度 → 重叠置信度
    # FrustumPosePredictor: 8顶点视锥 → 初始位姿 (XZ平面)
    # OverlapLoss: BCE + GT 视锥 mask
```

---

## 12. 工具与可视化

### 12.1 模型工厂

**文件**: `utils/model_factory.py` (178行)

```python
def create_model(config):
    # 根据 config.model.version 创建:
    # 'v1' → ICPoseNet
    # 'v2' → ICPoseNetV2
    # 'v2_lite' → ICPoseNetV2 (轻量)
```

### 12.2 损失工厂

**文件**: `utils/loss_factory.py` (289行)

```python
class CombinedLoss:
    # pose_loss + overlap_loss + diversity_loss + reprojection_loss
    # 可配置权重, 支持 C2F 和标准模式
```

### 12.3 训练工具

**文件**: `utils/training_utils.py` (389行)

```python
compute_relative_pose(T1, T2)     # 相对位姿
compose_pose(delta, T)             # 位姿组合
compute_pose_error(pred, gt)       # 旋转(°) + 平移(m) 误差
prepare_model_inputs(batch, cfg)   # v1/v2 输入格式转换
GradientClipper                    # 自适应梯度裁剪
EMATracker                         # 指数移动平均追踪器
MetricLogger                       # 训练指标日志
create_optimizer(model, cfg)       # 带参数分组的优化器
create_scheduler(optimizer, cfg)   # CosineAnnealing / StepLR
```

### 12.4 可视化

**文件**: `utils/visualization.py` (324行)

```python
visualize_attention_maps()              # 注意力权重热力图
visualize_2d3d_correspondence()         # 2D-3D 对应可视化
visualize_pose_prediction()             # 位姿预测 vs GT 可视化
visualize_query_features()              # 查询特征 PCA
visualize_feature_similarity_matrix()   # 特征相似度矩阵
```

---

## 13. 脚本索引

### 训练脚本 (核心)

| 脚本 | 说明 |
|------|------|
| `scripts/train_ms_flow.py` (887行) | ★ MSFlowPoseNet 主训练 |
| `scripts/train_corr_pose.py` | CorrPoseNet 训练 |
| `scripts/train_corr_pose_v2.py` | CorrPoseNet V2 训练 |
| `scripts/train_multiscale_ae.py` | AutoEncoder 压缩器训练 |
| `train_v3.py` | ICPoseNetV3 训练 |
| `train.py` | 通用训练入口 |

### 评估/诊断脚本

| 脚本 | 说明 |
|------|------|
| `scripts/eval_corr_pose.py` | CorrPoseNet 评估 |
| `scripts/eval_corrpose_iters.py` | 迭代次数消融 |
| `scripts/eval_appearance_diagnostic.py` | Appearance 网络诊断 |
| `scripts/ablation_5deg.py` | 5° 消融实验 |
| `scripts/debug_fda.py` | FDA 调试 |
| `scripts/diagnose_pipeline.py` | 管线诊断 |

### 特征相关脚本

| 脚本 | 说明 |
|------|------|
| `scripts/extract_features_v2.py` | V2 特征提取 |
| `scripts/extract_multiscale_features.py` | 多尺度特征提取 |
| `scripts/compute_pca.py` | PCA 降维 |
| `scripts/visualize_multiscale_features.py` | 多尺度特征可视化 |

### Shell 脚本 (58个)

主要为重训练实验脚本 (`auto_retrain*.sh`, `retrain*.sh`)，覆盖 OldHospital 数据集的多个训练配置。

---

## 14. SplatLoc 集成模块

**目录**: `splatloc_modules/`

### 14.1 FeatureDecoder

```python
# splatloc_modules/models/decoders.py
class FeatureDecoder(nn.Module):
    # HashGrid 位置编码 → MLP → L2 归一化特征
    # encoding: tinycudann Grid (Dense/Hash/SH/Freq/Identity)
    # FeatureNet: Multi-layer MLP (input_ch → hidden_dim → final_dim)
    # 场景空间归一化: pos → [0,1] via bounding_box
```

### 14.2 GaussianModel (标准 3DGS)

```python
# splatloc_modules/gaussian_splatting/scene/gaussian_model.py (680行)
class GaussianModel:
    # 标准 3DGS 模型 (from Inria GRAPHDECO)
    # 完整的: xyz + SH + scaling + rotation + opacity
    # 额外属性: marker, kp_score (SuperPoint keypoint)
    # Densification: clone + split + prune
    # 完整的 training_setup + optimizer 管理
```

### 14.3 渲染器

```python
# splatloc_modules/gaussian_splatting/gaussian_renderer/__init__.py
def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color):
    # 使用 diff_gauss (旧版 diff-gaussian-rasterization)
    # SH 颜色预计算 + keypoint score 拼接
```

---

## 15. Git 历史与演进

```
9e9a325 feat/fix: exp030-032 updates (iterative routing, config tuning, fine iters)
c218efa best results
3ad9cd2 FDA 900-frame eval complete
ec50462 FDA v2 optimization: chunk_size=128, align_fast v2, early stopping
8120a09 refactor: project cleanup and reorganization
400c8be feat: raw feature 3DGS embedding + AnyLoc VLAD place recognition
bf7deee Add detailed multi-scale visualization script
d6ff681 Improvements: cross-sequence retrieval validation + v2 accelerated training
d3eab51 Phase 3: DINO CLS Token FAISS place recognition
db8eb43 Phase 2: Multi-scale 3DGS feature embedding training
07ee2d1 improve: AE calibration + MSE loss + 100 epochs
c58c0ee fix: remove reflect-padding, crop SD UNet internal zero-pad instead
9625981 fix: 可视化subplot从4列改为5列(fine_sd+fine_dino+mid+coarse)
fa10b6f Phase1 fix: Fine层拆分为独立的SD s3/DINO + AutoEncoder替代PCA
1b1fa77 Phase1: 多尺度特征提取器 + PCA降维
22c1490 fix the feature extraction padding issues
f3025ea Add reprojection loss
3918640 Fix feature Selection
da66d89 first commit
```

### 当前状态

- **活跃分支**: `v2-iterative-routing`
- **修改文件**: `feature_3dgs/train_2dgs_geometry.py` (WildGaussians appearance 集成)
- **新增文件**: `scripts/auto_retrain38g.sh`
- **活跃实验**: exp031 (iterative refinement) → exp032 (cosine loss + fine_iters=8)
- **活跃训练**: OldHospital v3_retrain38 系列 (a-g) 2DGS 几何重建

---

## 附录: 配置文件索引

| 配置 | 关键参数 | 说明 |
|------|---------|------|
| `exp031_iterative.yaml` | fine_iters=4, outer_iters=3, lr=1e-4, epochs=80 | 迭代细化基线 |
| `exp032_cosine_fiters8.yaml` | fine_iters=8, rot_loss=cosine, trans_weight=10, lr=5e-5, epochs=60 | 余弦旋转损失 + 8次迭代 |
| `exp030_ms_flow_v2.yaml` | Multi-scale flow V2 | 多尺度光流改进 |
| `exp029_feature3dgs.yaml` | Feature 3DGS | 特征嵌入集成 |
| `exp028_gpu_optimized.yaml` | GPU 优化 | 训练加速 |
| `exp027_temperature_fix.yaml` | 温度参数修复 | 数值稳定性 |
| `exp026_fix_inproj.yaml` | 输入投影修复 | 维度对齐 |
| `exp025_synthetic_data.yaml` | 合成数据 | 数据增强 |
| `exp024_accuracy_improvement.yaml` | 精度改进 | 整体优化 |
| `exp023_heatmap_fix.yaml` | Heatmap 修复 | 特征图处理 |
| `exp022_arch_opt.yaml` | 架构优化 | 网络结构 |
| `exp021_baseline.yaml` | 基线实验 | 对照组 |
| `exp020_config.yaml` | 初始配置 | 起始点 |
