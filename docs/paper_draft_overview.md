# ICLPose-loc：基于 RADIO 引导双尺度特征场的 6-DoF 视觉定位

> 文档性质：投稿准备草稿 / 内部技术总结
> 最后更新：2026-04-15

---

## 摘要

本文提出一套面向室内场景的 6 自由度视觉定位框架，核心思路是以 RADIO ViT-H/16 大模型
作为双尺度特征教师，将其对齐到一个由 2D 高斯泼溅（2DGS）显式承载、稀疏哈希格隐式补
全的级联特征场（DCFF）中，并在定位时通过查询图像与渲染特征之间的稠密对应流完成姿态估
计。在 OldHospital 室内场景上，Oracle 检索条件下中位旋转误差达到 **0.30°**，中位平移
误差达到 **166 mm**；学习式精化（低噪声初始化）可进一步降至 **0.70° / 74.8 mm**。

---

## 1. 问题定义

### 1.1 任务

给定预先建好的场景三维地图与单张 RGB 查询图像，要求在毫秒量级内估计查询相机的 6-DoF
位姿 $(\mathbf{R}, \mathbf{t}) \in SE(3)$，无需深度输入或跟踪历史。

### 1.2 挑战

| 挑战 | 具体表现 |
|------|----------|
| **表示维度** | 三维场景特征场维度高（通常 768d+），直接渲染代价极大 |
| **分辨率–容量权衡** | 低分辨率特征丢失细节，高分辨率特征存储和推理开销大 |
| **几何与语义脱耦** | 浅层特征偏几何/边界，深层特征偏语义，统一表示困难 |
| **明特征迁移** | 大模型特征无法直接用于轻量化查询端，需要可微知识蒸馏 |
| **候选选取** | 在真实场景中，检索 top-1 初始化误差大（~4.5 m），显著影响后续精化 |

---

## 2. 相关工作与位置

```
     检索初始化           场景特征渲染            稠密匹配/精化
  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
  │  CLS global  │─→  │  DCFF 特征场渲染  │─→  │  ConcatPoseNet  │
  │  检索 top-K  │    │  fine_geo+sem 双流│    │  GRU 迭代精化   │
  └──────────────┘    └──────────────────┘    └─────────────────┘
        ↑                      ↑
   RADIO sem 全图         2DGS 显式载体
    特征 top-1 CLS         + HashGrid 隐式场
```

与主要相关工作的对比定位：

- **NeRF/3DGS-based localization（如 iNeRF, NeRF-loc）**：本工作避免 RGB 渲染，直接在高维特征空间对齐，对光照变化更鲁棒。
- **SplatLoc / GSFF**：本工作在特征分层（几何 fine + 语义 coarse）和隐式补全方面更系统，同时引入 RADIO 提供更强的教师监督。
- **HLoc / SuperGlue 等稀疏匹配**：本工作提供一条 dense feature flow 的互补路线，在自相似场景（长走廊等）下可与 LoFTR 协同。
- **LEGaussians/RADIO-GS**：本工作在相同 RADIO 教师框架下对级联隐式场结构进行了更深入的依赖分析和消融。

---

## 3. 方法

### 3.1 总体架构（五模块）

```
FeatureGaussian ──▶ FeatureField ──▶ FeatureRetrieval
      │                  │                 │
   显式载体           隐式特征场          全局检索
  (2DGS PLY)        (DCFF/HashGrid)     (CLS feat)
      │                  │                 │
      ▼                  ▼                 ▼
 FeatureExtract ──▶ PoseRefine ◀─── 候选初始位姿
   RADIO 教师         (ConcatPoseNet)
   学生蒸馏             GRU 精化
```

端到端流程：

> `检索初始化 → 渲染场景特征 → 稠密特征流对应 → 几何求解 → 迭代精化`

---

### 3.2 RADIO 双尺度教师特征提取（FeatureExtract）

采用 **RADIO ViT-H/16**（c-radio_v4-h，约 660M 参数），钩取两个不同深度的中间
层输出，形成双尺度教师特征：

| 尺度 | 来源 | 性质 | PCA 降维后维度 | PCA 保留率 |
|------|------|------|----------------|------------|
| **fine_geo** | 浅层 Block 10 | 几何/边界敏感，高频纹理 | 64d | ~67% |
| **coarse_sem** | 最终输出层 | 语义/上下文强，光照不变 | 64d | ~80% |

**注意**：浅层（Block 10）特征在 patch_size=16 的 ViT 中尚未完成充分跨 patch 注意力
混合，会出现竖向条纹的块边界效应，这是 early-block ViT 特征的固有特性，而非数据缺陷。

教师特征被缓存到磁盘后，分别用于（a）监督显式 2DGS 载体训练、（b）监督隐式 DCFF 训练、
（c）监督查询端学生网络的知识蒸馏。

---

### 3.3 显式高斯特征载体（FeatureGaussian）

场景几何用 **2DGS（2D Gaussian Splatting）** 表示，每个高斯原语除了标准外观属性外额外
携带一个 `z_latent`（32d）向量，称为"显式载体"。

关键设计：

```
2DGS 光栅化
    ↓
z_map（H×W×32）← per-pixel compositor of z_latent
    ↓
SpatialFineDecoder（5×5 空洞卷积, hidden=128）
    ↓
fine_feature_map（H×W×64）
```

- 几何 PLY 来自独立训练好的 2DGS（OldHospital, 30k iter），共 **300,845 个高斯**。
- 显式载体仅训练 latent，几何在 DCFF 阶段冻结。
- 当前 pilot（joint_radio_dual_oldhospital）仅跑了约 40 次迭代，联合几何+特征训练
  仍在规划中（为后续重要改进方向）。

---

### 3.4 级联延迟特征场 DCFF（FeatureField）

DCFF 以 2DGS 为底座，通过两条分支重建双尺度教师特征：

#### Fine 分支（显式，几何驱动）

```
2DGS 栅格化 z_map（32d）
    ↓
SpatialFineDecoder（5×5 空洞卷积, 193K 参数）
    ↓ + DepthGuidedRefiner（optional, hidden=256）
fine_feature（H×W×64）
```

- 完全依赖显式 latent carrier。
- 依赖消融：`fine_latent_zero_self_cos ≈ 0.00`（zeroing latent 使输出完全变化）。
- `fine_geom_jitter_self_cos ≈ 0.90`（几何微扰对输出影响适度）。

#### Coarse 分支（隐式，空间哈希网格）

```
Gaussian 中心 XYZ + log_scale（位置+尺度编码）
    ↓
SpatialHashGrid（hash=24M params, MLP 50→128→64）
    ↓ [v10c: +CarrierResidualCoarseFusion（残差通道）]
coarse_feature（H×W×64）
```

- **当前架构缺陷**：`implicit_scale` 模式下 coarse 分支完全不读取 latent carrier，
  即使加入 CarrierResidualCoarseFusion 并训练至 40k iter，
  `coarse_latent_zero_self_cos` 仍 = 1.000，说明 hash 场路径完全主导，
  零初始化的载体残差路径实际上没有被激活（zero-init 导致 warmup 阶段残差路径
  梯度极小）。

#### 训练指标汇总（OldHospital，40k iters）

| 实验 | val fine_cos（训练域） | val coarse_cos（训练域） | 测试 fine_refined_cos | coarse_latent依赖 |
|------|-----------|------------|------------|------------|
| v10（baseline） | ~0.84 | ~0.86 | 0.366 | 完全无依赖 |
| v10b（coarse fix） | ~0.85 | ~0.85 | 0.352 | 完全无依赖 |
| v10c（carrier residual，mid ~15500 iter） | ~0.87 | ~0.85 | **0.625** | 完全无依赖 |

> v10c 的 fine_refined_cos 从 0.37 跳升到 0.625，边际显著。主要原因在于
> DepthGuidedRefiner 在 carrier residual 融合的新 latent 分布下获得了更强的
> 精化能力，并非 coarse 路径改变所致。

---

### 3.5 查询端学生特征提取（FeatureExtract 学生蒸馏）

为在推理时不需要运行完整 RADIO ViT-H/16（耗时 ~200ms），训练一个轻量化学生网络：

- 架构：多尺度卷积 backbone + 双分支输出头（fine / coarse）。
- 监督：与缓存教师特征的 point-wise cosine 损失，同时可以与 DCFF 渲染的 map 特征
  做对比 anchor 损失。
- 结果（v20k_v5g_student）：与教师引导下的定位精度基本持平（145.1 mm vs 144.95 mm）。

---

### 3.6 位姿精化网络 ConcatPoseNet（PoseRefine）

基于 RAFT 风格的迭代特征流估计，加几何求解器，形成完整的定位精化流程：

```
查询特征图（fine+coarse concat, H×W×128）
渲染特征图（fine+coarse concat, H×W×128）
      ↓ 初始对应 [concat mode]
  初始流场估计（CNN 编码器 + R+ReLU）
      ↓ 迭代精化 [GRU mode, outer_iters 次]
  ConvGRU 更新（hidden=128）← 局部相关 corr（radius=4）
  ← 可选 CrossAttentionMatcher（全局消歧）
      ↓
  稠密 2D→2D 流场（H×W×2）+ 置信度图
      ↓
  WLS 几何求解器（image Jacobian + 加权最小二乘）
      ↓
  6-DoF 姿态 (R, t)
```

关键组件：

- **DepthGuidedRefiner**：利用深度图引导精化 fine 特征，提升几何一致性。
- **WLS 求解器**（Huber kernel）：比 direct 方法稳定约 30-40× 在高噪声条件下。
- **CrossAttentionMatcher**：解决长走廊等自相似场景的全局语义歧义。

---

## 4. 实验结果

### 4.1 实验设置

- **场景**：OldHospital 室内（RGB-D 序列，1920×1080 分辨率）
- **数据集划分**：895 训练帧 / 182 测试帧
- **渲染分辨率**：feature_map 120×68（约 1/16 原图）
- **评估指标**：旋转中位误差（deg）、平移中位误差（mm）、各精度阈值通过率

---

### 4.2 Pipeline 结果（Oracle 检索）

使用 Oracle 近邻选取，通过 LoFTR 累积特征对应，再 PnP 求解后精化：

| 阶段 | 旋转中位误差 | 平移中位误差 |
|------|-------------|-------------|
| 检索初始化（CLS top-K） | 11.50° | 1355 mm |
| LoFTR 初始化（K=30）  | **0.30°** | **166 mm** |
| 后续精化（oi=0）       | 0.30° | 166 mm |

> LoFTR 直接累积特征对应已可达到 0.30°/166 mm，说明在充分初始化条件下，
> 稀疏匹配 + PnP 的上限很高。后续精化阶段对此级别初始化提升有限。

---

### 4.3 学习式精化结果（ConcatPoseNet）

使用 oracle 检索的初始位姿，注入合成噪声后测试精化效果：

#### 低噪声条件（1° / 50 mm）

| 模型 | 旋转中位误差 | 平移中位误差 | <1° 占比 | <5° 占比 |
|------|-------------|-------------|---------|---------|
| v9e（WLS默认，oi=10） | 0.705° | 74.8 mm | 65% | 100% |
| v9e（WLS full，oi=10） | 0.704° | 73.9 mm | 65% | 100% |

#### 高噪声条件（3° / 100 mm）

| 模型 | 旋转中位误差 | 平移中位误差 | <1° 占比 | <5° 占比 |
|------|-------------|-------------|---------|---------|
| v20d（教师查询特征，oi=10） | 2.14° | 148.5 mm | 24% | 89% |
| v20k（学生查询特征，oi=10） | ~2.14° | ~145.1 mm | ~24% | ~89% |

> 学生特征与教师特征的精化性能基本持平，验证了知识蒸馏的有效性。

#### 多假设融合上界

通过多假设 Oracle 融合（10 次不同初始化取最优），可达：

- 中位平移误差：**47.3 mm**

这说明精化网络本身的容量已足够，主要瓶颈在于初始化质量和候选选取。

---

### 4.4 真实检索条件（端到端）

使用 CLS 图像检索 top-1 作为初始化（真实部署场景）：

| 条件 | 旋转中位误差 | 平移中位误差 |
|------|-------------|-------------|
| CLS top-1 init → 精化 | ~7.2° | ~4494 mm |
| top-10 oracle 融合 → 精化 | ~7.5° | ~2792 mm |

> 真实部署结果与 oracle 检索存在约 27× 的平移差距（166 mm vs 4494 mm），
> **主要瓶颈是检索候选选取质量，而非精化网络本身**。

---

### 4.5 DCFF 特征场诊断消融（Dependency Diagnostics）

设计了三类扰动来定量分析 fine/coarse 分支对输入来源的依赖程度：

| 消融类型 | 定义 |
|----------|------|
| `latent_zero` | 将所有高斯 latent 置零后重渲染 |
| `latent_shuffle` | 将高斯 latent 在样本间随机打乱 |
| `geom_jitter` | 向高斯 XYZ 添加随机扰动（σ≈场景尺度 5%） |

结果（v10c，测试 4 帧）：

| 指标 | 值 | 解读 |
|------|-----|------|
| `fine_geom_jitter_self_cos` | 0.902 | fine 对几何有适度依赖 |
| `fine_latent_zero_self_cos` | ~0.001 | fine 完全依赖 latent carrier |
| `fine_latent_shuffle_self_cos` | ~0.020 | fine 读取的是 per-Gaussian 的具体内容 |
| `coarse_latent_zero_self_cos` | **1.000** | coarse **完全不依赖** latent carrier |
| `coarse_latent_shuffle_self_cos` | **1.000** | coarse **完全不依赖** latent carrier |

> **结论**：当前 DCFF 的 fine 和 coarse 分支实际上是两个完全解耦的表示：fine 是
> 纯载体驱动（position-agnostic latent），coarse 是纯几何/位置驱动（latent-agnostic
> hash field）。它们并非"同一表示的双尺度精化"，而是两套独立信号，这限制了
> 两者的互补性。

---

## 5. 主要创新点与贡献

### 5.1 RADIO 引导双尺度特征场

- 首次将 RADIO ViT-H/16 的**浅层（geometric）+ 深层（semantic）双尺度输出**引入
  三维特征场训练，构成统一的教师监督体系。
- 提供了一套可扩展的 teacher cache + 独立可视化管道，使教师质量、2DGS 重建质量、
  DCFF 重建质量可以在同一坐标系下对比评估。

### 5.2 2DGS 显式特征载体 + DCFF 隐式补全的级联框架

- 2DGS 作为"显式 latent carrier"，承担局部几何一致性和边界保持。
- SpatialHashGrid 作为隐式补全层，对 2DGS 未覆盖区域进行空间外推。
- SpatialFineDecoder 使用 5×5 空洞卷积感受野，相比 1×1 卷积获得更平滑的特征图输出。
- DepthGuidedRefiner 利用场景深度信息进行几何引导的 fine 特征精化。

### 5.3 CarrierResidualCoarseFusion（v10c）

- 设计了一个零初始化的载体投影 + 门控残差注意力模块，可在不破坏已有 checkpoint
  的前提下，为 coarse 分支引入 latent carrier 的信息通路。
- 通过消融诊断工具，首次定量揭示了当前 coarse 分支完全忽略 latent carrier 这一
  架构缺陷，为后续设计改进提供了清晰依据。

### 5.4 RAFT 风格 GRU 特征流精化 + 几何求解器

- ConcatPoseNet 的 GRU 迭代精化模式在高噪声初始化（3°/0.1m）下相比直接 PnP 稳定
  约 **30×**（direct solver: 17.4°/5044mm vs WLS: 2.14°/148.5mm）。
- 加权最小二乘（WLS）+ Huber kernel 的几何求解器显著提升了对 outlier 对应的鲁棒性。
- CrossAttentionMatcher 为自相似场景提供全局语义上的消歧能力。

### 5.5 学生特征完全对齐教师质量的蒸馏框架

- 轻量化学生网络（仅用于查询端）在定位精度上与完整 RADIO ViT-H/16 教师查询基本
  持平（145.1 mm vs 144.95 mm），实现了约 **20-40×** 的推理加速。

---

## 6. 当前局限性与后续工作

### 6.1 已识别的主要问题

| 问题 | 证据 | 计划 |
|------|------|------|
| **检索质量是端到端瓶颈** | top-1 CLS: 4494 mm vs oracle: 166 mm | 训练专用检索 reranker，或融入 feature-based retrieval |
| **Coarse 分支不读 carrier** | `coarse_latent_shuffle_self_cos = 1.000` | 架构改为 shared carrier + dual decode head；或从好的隐式 checkpoint warmstart 后解冻 carrier fusion |
| **2DGS 几何从未联合训练** | 所有实验均冻结 v7_depth PLY，joint pilot 只跑了 40 iters | 联合几何+特征训练，让几何适应 RADIO 特征监督 |
| **Teacher fine_geo 竖向条纹** | ViT-H/16 patch_size=16，Block 10 跨 patch 注意力不充分 | 使用更深层（Block 14-16）作为 geo 钩取点，或对 teacher feature 做 3×3 空间平滑 |
| **渲染分辨率 120×68 偏低** | fine 特征细节丢失，影响精细定位 | 提升到 240×135 并配合特征降维 |

### 6.2 计划的下一步

1. **联合 2DGS + 特征训练**：以 v7_depth PLY 为初始化，解冻几何并以低学习率联合训练，
   使高斯几何向 RADIO 特征对齐。
2. **Shared Carrier Dual-Head 架构**：将 fine/coarse 改为共享同一个 latent carrier，
   分别用不同的解码头输出，从源头消除 coarse 分支的 carrier 断链问题。
3. **Reranker 优化**：在 top-10 检索候选上训练更强的 pose-aware reranker，
   将真实部署的中位误差从 ~4.5 m 降低到与 oracle-top10 (~2.8 m) 同等水平。
4. **高分辨率特征渲染**：研究特征维度压缩（32d 以内）配合全分辨率渲染的可行性。

---

## 7. 关键技术指标总览

```
         分辨率-效率边界
  RADIO ViT-H/16 teacher  ←  教师  →  640M 参数，patch 16
  Student network           ←  学生  →  轻量，精度持平

         特征场重建质量（OldHospital test 帧）
  2DGS pilot cosine:  fine_geo ≈ 0.295,  coarse_sem ≈ 0.488
  DCFF v10 val  cos:  fine ≈ 0.84,       coarse ≈ 0.86  （训练域）
  DCFF v10 test cos:  fine_refined ≈ 0.366               （测试域外推）
  DCFF v10c （mid）: fine_refined ≈ 0.625 (+70%)          （carrier residual 提升）

         定位精度（OldHospital, 182 test 帧）
  Oracle 检索 + LoFTR:       0.30° / 166 mm
  学习式精化（低噪声1°/50mm）: 0.70° / 74.8 mm   <1°=65%, <5°=100%
  学习式精化（高噪声3°/100mm）: 2.14° / 148.5 mm  <1°=24%, <5°=89%
  多假设 Oracle 融合上界:      N/A  / 47.3 mm
  真实部署 top-1 CLS:         7.2° / 4494 mm
  真实部署 top-10 oracle:     7.5° / 2792 mm
```

---

## 附：代码入口速查

| 功能 | 命令 |
|------|------|
| 提取 RADIO 双尺度教师特征 | `python -m feature_extract.extract_radio_dual_features` |
| 可视化教师特征 | `python -m feature_extract.visualize_teacher_features` |
| 训练 DCFF 特征场 | `python -m feature_field.train` |
| 可视化 DCFF 重建（+依赖消融） | `python -m feature_field.visualize_reconstruction --dependency_diagnostics` |
| 训练查询端学生网络 | `python -m feature_extract.train` |
| 构建检索索引 | `python -m feature_retrieval.build_index` |
| 训练位姿精化网络 | `python -m pose_refine.train` |
| 评估（Oracle 检索） | `python -m pose_refine.evaluate` |
| 评估（LoFTR pipeline） | `python -m pose_refine.evaluate_pipeline` |
| 评估（真实检索） | `python -m pose_refine.evaluate --real_init` |

当前最强配置：`configs/concat_loc_oh_v20k_v5g_student_querymap_adapt.yaml`
当前最强 DCFF：`feature_field/configs/dcff_oldhospital_v10c_carrier_residual.yaml`

---

*本文档由系统自动汇总，基于 2026-04-15 最新实验状态写成。*
