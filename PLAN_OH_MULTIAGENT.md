# ICLPose OldHospital 专攻计划 — 多Agent并行执行

> 目标场景：Cambridge Landmarks OldHospital（室外，50×40m，895 train / 182 test, 跨序列测试）  
> 当前最优：3.24° / 1300mm（exp142, 仅4 epoch）  
> SOTA参考：ACE 0.32° / 220mm（scene coordinate regression + RANSAC+PnP）  
> 硬件：6×RTX 4090，每agent一张卡  
> 不做：FDA、Room0/Stairs优化

---

## 当前管线诊断

```
离线提特征 (SD 1280d + DINO 768d)
  ↓ AE压缩 (→ 32/64/64/64d per scale)
  ↓ 3DGS embedding (300K Gaussians × 224d, L1+cosine loss)
  ↓ 在线渲染 (per-scale渲染 → 特征图)
  ↓ MSFlowPoseNet (correlation → flow → Image Jacobian → pose)
```

### 根本问题分析

| 问题 | 证据 | 影响 |
|:---|:---|:---|
| **特征embedding loss停滞** | 训练loss从0.060仅降到0.053 (15K iters), 改善<12% | 3DGS无法忠实重建特征，渲染vs真值差距大 |
| **AE压缩信息丢失** | SD 1280d→512d→64d, 两级压缩, 无任务反馈 | 压缩后特征可能丢失定位关键信息 |
| **特征通道噪声** | 224d全部参与correlation，无通道选择 | 噪声通道污染correlation volume |
| **correlation局部窗口** | r=4仅覆盖±4像素 = ±8° at fine scale | 大位移时完全匹配失败 |
| **过拟合严重** | OH所有实验best在epoch 0-5, 100ep后退化到5-6° | 模型容量vs数据量不匹配 |
| **跨序列测试** | train=seq{1-3,5-7,9}, test=seq{4,8}, 平均距离2.1m | 测试视角显著偏离训练分布 |

---

## 三大子任务定义

### Task 1: 高斯重建优化

**目标**: 提升3DGS几何+外观质量，为特征embedding提供更好的几何基础

**当前状态**:
- 2DGS v7_depth, 30K iters, 300,845 Gaussians
- 从COLMAP sparse points初始化

**改进方向**:
1. **3DGS vs 2DGS对比**: 2DGS理论上表面更好但对大场景可能不如3DGS
2. **增加densification iterations**: OH场景大(50×40m)，30K iter可能不够
3. **深度监督**: 利用mono_depth约束几何
4. **抗锯齿**: 使用mip-splatting避免远处Gaussian的aliasing
5. 重建指标：PSNR/SSIM on held-out views

**验证标准**: 新重建的3DGS在hold-out view上PSNR > 旧值

### Task 2: 特征Embedding与Selection

**目标**: 将定位有效特征嵌入Gaussians，同时通过通道级选择剔除噪声

**当前状态**:
- 原始特征: SD s3(640d) + SD s4(1280d) + SD s5(1280d) + DINO(768d)
- AE压缩后: fine_sd(64) + fine_dino(64) + mid(64) + coarse(32) = 224d per Gaussian
- Embedding: L1 + cosine loss, loss仅从0.06→0.053, 停滞

**核心改进**:

#### 2a. 通道级特征选择 (Channel Selection)

**动机**: 224d中不是所有通道都对定位有帮助。SD特征包含大量语义/风格信息，对定位是噪声。

**方案**: 在3DGS embedding阶段引入可学习通道门控：
```python
class ChannelGate(nn.Module):
    """Per-scale learnable channel importance gating"""
    def __init__(self, n_channels):
        self.gate_logits = nn.Parameter(torch.zeros(n_channels))  # sigmoid门控
    
    def forward(self, features):
        # features: [N_gaussians, C] or [C, H, W]
        gates = torch.sigmoid(self.gate_logits)  # [C], 值在[0,1]
        return features * gates
```

- 每个scale独立一个ChannelGate
- 训练时用L1 sparsity正则鼓励稀疏
- 训练收敛后，gate<0.1的通道可直接裁剪 → 降维
- 关键：门控应在embedding loss和/或下游pose loss中联合训练

#### 2b. 高维直接Embedding (跳过AE压缩)

**动机**: AE压缩(1280→64d)信息丢失不可控。如果Gaussian能存更高维特征，可能更好。

**方案**: 
- 试验embedding维度: 224d → 384d → 512d
- 直接从原始SD特征(不经AE)提取
- 在3DGS embedding中引入轻量MLP decoder替代固定维度slice
- 权衡: 更高维 = 更多显存 + 更慢渲染，但损失更少信息

#### 2c. 任务驱动Embedding (Task-Driven)

**动机**: 当前embedding loss (L1+cosine) 优化的是特征重建保真度，不是定位精度。

**方案**:
- 在embedding训练中引入定位代理loss
- 方法1: 渲染两个近邻视角的特征图 → 计算mutual nearest neighbor匹配率
- 方法2: 渲染特征 → 冻结的简化pose回归head → pose proxy loss
- 方法3: 对比学习 — 同一3D点在不同视角渲染的特征应一致(正对), 不同点应不同(负对)

### Task 3: 位姿估计网络

**目标**: 改进flow匹配和pose求解，充分利用Task 2提供的更好特征

**当前状态**:
- MSFlowPoseNet: coarse correlation → mid guided → fine GRU → Image Jacobian solver
- correlation radius r=4 (±4 pixels)
- 3 outer iterations (train), 10 (val)

**改进方向**:

#### 3a. Coarse阶段Attention替代Global Correlation
- coarse分辨率15×26 = 390 tokens, attention 390×390矩阵完全可行
- Cross-attention自带位置加权，无需手工correlation
- 参考: GMFlow, LoFTR

#### 3b. 增大搜索窗口
- 当前r=4, 在fine scale(69×121)上仅覆盖±4/121 ≈ ±3.3%视野
- 增大到r=8或r=12, 覆盖更大位移
- 权衡: 显存线性增长

#### 3c. 先验引导的搜索 (Coarse-to-Fine传递)
- coarse阶段的flow可以指导fine阶段的搜索中心
- 当前只是简单插值，可以做deformable search

### Task 3*: 多尺度特征使用策略

**当前问题**: coarse(15×26), mid(30×53), fine(69×121) 分辨率跨度大，特征来自不同backbone层

**改进方向**:
- 在pose网络内部做尺度间信息传递 (FPN-style top-down)
- 特征对齐: 确保不同scale的decode_dim相同，使cross-scale correlation有意义
- Scale-adaptive weighting: 根据noise级别动态调整各scale权重

---

## Agent分工

### Agent 0 (GPU 0): 3DGS重建优化 [Task 1]

**目标**: 训练更好的几何重建作为embedding基础

**步骤**:
1. 评估当前2DGS v7_depth质量 (渲染held-out view, 计算PSNR/SSIM)
2. 用gsplat重训3DGS (更多iterations: 50K, 更激进densification)  
3. 加入深度监督(mono_depth loss)
4. 对比2DGS vs 3DGS重建质量

**输出**: 最优geometry的PLY文件 (point_cloud.ply) + 训练log

**成功标准**: Hold-out PSNR提升 > 1dB

### Agent 1 (GPU 1): 通道选择Embedding [Task 2a]

**目标**: 在现有3DGS geometry上训练带通道门控的feature embedding

**步骤**:
1. 在 `feature_3dgs/multiscale_gaussian_model.py` 中添加 `ChannelGate` 模块
2. 修改 `feature_3dgs/train_multiscale_embedding_v2.py`:
   - 在embedding训练loss中加入gate sparsity正则
   - 记录每个通道的gate值变化
3. 训练OH embedding with channel selection
4. 分析哪些通道被gate关闭 → 输出通道重要性排名
5. 用selected channels重训embedding → 验证降维后精度

**输出**: 
- 带channel gate的per-scale模型
- 通道重要性分析报告
- 降维后的新嵌入模型

**成功标准**: 
- 自动识别出 > 20% 低重要性通道
- 剔除后embedding loss不退化

### Agent 2 (GPU 2): 高维直接Embedding [Task 2b]

**目标**: 跳过AE压缩，直接将更高维特征嵌入Gaussians

**步骤**:
1. 提取OH原始SD/DINO特征 (不经AE压缩)
   - fine_sd: 640d (SD s3), fine_dino: 768d
   - mid: 1280d (SD s4), coarse: 1280d (SD s5)
2. 训练高维embedding:
   - 方案A: 全维度 (640+768+1280+1280=3968d) — 可能太大，先PCA到512
   - 方案B: 每scale PCA到更大维度 (fine_sd: 128, fine_dino: 128, mid: 128, coarse: 64 = 448d)
   - 方案C: 用轻量MLP在Gaussian端decode (Gaussian存64d latent, MLP decode到各scale维度)
3. 评估: 渲染质量 + 下游pose精度

**输出**: 高维/替代维度的embedding模型

**成功标准**: Embedding loss < 0.050 (现有0.053)

### Agent 3 (GPU 3): Pose网络改进 [Task 3]

**目标**: 改进MSFlowPoseNet的匹配和求解

**步骤**:
1. 实现 `CrossAttentionMatcher` 替代coarse global_correlation
2. 测试增大 local_radius: r=4→8→12
3. 实现coarse→fine flow传递 (guided search center)
4. 用现有embedding和当前best checkpoint (exp142) 验证

**输出**: 改进后的MSFlowPoseNet + 对比实验

**成功标准**: OH rot < 2.8° (突破3.24°天花板)

### Agent 4 (GPU 4): 集成验证 [Task 2+3 组合]

**目标**: 将各agent产出组合，验证端到端效果

**步骤**:
1. 等待Agent 1/2/3各自产出
2. 组合最优embedding + 最优pose网络
3. 运行完整训练 (100 epochs)
4. 与基线对比

**触发条件**: Agent 1-3至少有一个产生有效改进后启动

### Agent 5 (GPU 5): 对比学习Embedding [Task 2c]

**目标**: 用对比学习/匹配代理loss训练更好的特征embedding

**步骤**:
1. 设计viewpoint-consistent contrastive loss:
   - 渲染两个相邻视角 → 3D点对应已知(Gaussian对应关系)
   - 同一Gaussian在两视角渲染的特征应一致(正对)
   - 不同Gaussian的特征应不同(负对)
2. 或设计matching proxy loss:
   - 渲染query和reference特征 → 计算correlation → 验证GT flow对应处是否为max
3. 训练embedding with contrastive loss

**输出**: 对比学习训练的embedding模型

**成功标准**: 跨视角特征匹配率 > 当前baseline

---

## 执行时间线

```
╔══════════════════════════════════════════════════════════════════╗
║ Phase 1: 并行启动 (各agent独立)                                ║
║                                                                ║
║ GPU 0 ─── Agent 0: 3DGS重建 ─────────────────────────→ PLY    ║
║ GPU 1 ─── Agent 1: 通道选择Embedding ──────────────→ Gate模型  ║
║ GPU 2 ─── Agent 2: 高维直接Embedding ──────────────→ 嵌入模型  ║
║ GPU 3 ─── Agent 3: Pose网络改进 ───────────────────→ 网络改进  ║
║ GPU 5 ─── Agent 5: 对比学习Embedding ──────────────→ 嵌入模型  ║
║                                                                ║
║ Phase 2: 集成 (等Phase 1 best结果)                             ║
║                                                                ║
║ GPU 4 ─── Agent 4: 最优Embedding + 最优Pose网络 → 端到端验证   ║
║                                                                ║
║ Phase 3: 迭代优化                                              ║
║                                                                ║
║ 根据Phase 2结果，聚焦最有效方向，释放GPU给最有价值的agent      ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 关键技术实现细节

### 通道门控 (Agent 1)

```python
# feature_3dgs/channel_gate.py
import torch
import torch.nn as nn

class ChannelGate(nn.Module):
    def __init__(self, n_channels, init_bias=2.0):
        super().__init__()
        # 初始化为正值，sigmoid后接近1 → 初始不丢弃任何通道
        self.gate_logits = nn.Parameter(torch.full((n_channels,), init_bias))
    
    def forward(self, x):
        """x: [C, H, W] or [N, C]"""
        gates = torch.sigmoid(self.gate_logits)
        if x.dim() == 3:  # [C, H, W]
            return x * gates[:, None, None]
        else:  # [N, C]
            return x * gates[None, :]
    
    def get_importance(self):
        """返回通道重要性 [C]"""
        return torch.sigmoid(self.gate_logits).detach()
    
    def sparsity_loss(self, target_sparsity=0.3):
        """L1正则鼓励gate趋向0"""
        gates = torch.sigmoid(self.gate_logits)
        return torch.abs(gates.mean() - (1 - target_sparsity))
```

在embedding训练中:
```python
total_loss = recon_loss + sparsity_weight * sum(gate.sparsity_loss() for gate in gates_per_scale)
```

### CrossAttentionMatcher (Agent 3)

```python
# ic_models/cross_attention_matcher.py
class CrossAttentionMatcher(nn.Module):
    def __init__(self, dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=n_heads, 
                dim_feedforward=dim*4, batch_first=True
            ) for _ in range(n_layers)
        ])
    
    def forward(self, query_feat, ref_feat):
        """
        query_feat: [B, C, Hq, Wq]
        ref_feat: [B, C, Hr, Wr]
        Returns: correlation volume [B, Hr*Wr, Hq, Wq]
        """
        B, C, Hq, Wq = query_feat.shape
        Hr, Wr = ref_feat.shape[2:]
        
        # Flatten to tokens
        q = query_feat.flatten(2).permute(0, 2, 1)  # [B, Hq*Wq, C]
        k = ref_feat.flatten(2).permute(0, 2, 1)    # [B, Hr*Wr, C]
        
        # Cross-attention layers
        for layer in self.layers:
            q = layer(q, k)
        
        # Output correlation: dot product between attended query and ref
        corr = torch.einsum('bic,bjc->bij', q, k)  # [B, Hq*Wq, Hr*Wr]
        corr = corr.permute(0, 2, 1).view(B, Hr*Wr, Hq, Wq)
        
        return corr
```

### 匹配代理Loss (Agent 5)

```python
def matching_proxy_loss(rendered_feat_1, rendered_feat_2, gt_flow_1to2):
    """
    对比两个视角渲染的特征 + 已知对应关系
    rendered_feat_1: [C, H, W] from view 1
    rendered_feat_2: [C, H, W] from view 2
    gt_flow_1to2: [2, H, W] GT optical flow (from depth + poses)
    """
    # Warp feat_2 to feat_1's frame using GT flow
    warped_feat_2 = grid_sample(rendered_feat_2, gt_flow_1to2)
    
    # Positive loss: matched pixels should have similar features
    pos_sim = F.cosine_similarity(rendered_feat_1, warped_feat_2, dim=0)  # [H, W]
    pos_loss = (1 - pos_sim).mean()
    
    # Negative loss: random pixels should be dissimilar (hard negative mining)
    neg_shift = torch.roll(rendered_feat_1, shifts=5, dims=2)
    neg_sim = F.cosine_similarity(neg_shift, warped_feat_2, dim=0)
    neg_loss = F.relu(neg_sim - 0.1).mean()  # margin 0.1
    
    return pos_loss + 0.5 * neg_loss
```

---

## 文件修改清单

| Agent | 文件 | 操作 | 描述 |
|:---:|:---|:---:|:---|
| 1 | `feature_3dgs/channel_gate.py` | 新建 | 通道门控模块 |
| 1 | `feature_3dgs/multiscale_gaussian_model.py` | 修改 | 集成ChannelGate |
| 1 | `feature_3dgs/train_multiscale_embedding_v2.py` | 修改 | 加gate loss + 通道分析 |
| 2 | `scripts/extract_raw_features.py` | 新建 | 提取未压缩原始特征 |
| 2 | `feature_3dgs/train_highdim_embedding.py` | 新建 | 高维embedding训练 |
| 3 | `ic_models/cross_attention_matcher.py` | 新建 | Cross-attention coarse匹配 |
| 3 | `ic_models/ms_flow_pose_net.py` | 修改 | 集成attention matcher |
| 5 | `feature_3dgs/contrastive_embedding.py` | 新建 | 对比学习embedding |
| 4 | `configs/exp_oh_integrated.yaml` | 新建 | 集成实验配置 |

---

## 评估协议

### 统一评估脚本
```bash
python scripts/eval_iterative.py \
    --config configs/exp_XXX.yaml \
    --checkpoint output/exp_XXX/checkpoints/best.pth \
    --iters 10 \
    --output output/exp_XXX/eval_results.json
```

### 指标
| 指标 | 说明 | 目标 |
|:---|:---|:---|
| rot_mean | 旋转均值误差(°) | < 2.5° |
| rot_median | 旋转中位数误差(°) | < 2.0° |
| trans_mean | 平移均值误差(mm) | < 800mm |
| <1° | 旋转<1°比例 | > 10% |
| joint@1°/50mm | 联合指标 | > 5% |
| flow_epe | Flow端点误差 | < 4.0 |

### 对比基线
- exp142: dino_all_scales, 3.24°/1300mm (当前BEST)
- 每次改进必须报告 Δrot 和 Δtrans

---

## 风险与缓解

| 风险 | 缓解 |
|:---|:---|
| 通道门控退化为全1或全0 | 初始bias=2.0 + sparsity正则force 30%稀疏 |
| 高维embedding显存不足 | 渲染时分chunk (已有128d/chunk机制) |
| Attention coarse无效 | 保留correlation作为fallback，用config切换 |
| 对比学习训练不稳定 | 温度系数τ从0.1→0.05 anneal + gradient clip |
| Agent间依赖阻塞 | Phase 1各agent完全独立，无依赖 |

---

## 快速启动命令

```bash
# 先杀Phase 1旧实验
kill $(pgrep -f train_ms_flow) 2>/dev/null

# Agent 0: 3DGS重建 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python feature_3dgs/train_3dgs_geometry.py \
    --colmap_dir dataset/OldHospital/sparse/0 \
    --image_dir dataset/OldHospital \
    --output output/3dgs_oldhospital_v1 \
    --iterations 50000 &

# Agent 1: 通道选择Embedding (GPU 1)  
CUDA_VISIBLE_DEVICES=1 python feature_3dgs/train_multiscale_embedding_v2.py \
    --ply output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
    --feature_dir output/features_multiscale_compressed/OldHospital_indexed \
    --output output/feature_3dgs/oldhospital_channel_gate \
    --channel_gate --gate_sparsity_weight 0.01 &

# Agent 3: Pose网络 (GPU 3) — 先实现attention matcher再训练
CUDA_VISIBLE_DEVICES=3 python scripts/train_ms_flow.py \
    --config configs/exp_oh_attention.yaml \
    --warmstart output/exp142_oh_dino_all_scales/checkpoints/best.pth &
```
