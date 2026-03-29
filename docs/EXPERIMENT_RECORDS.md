# ICLPose OldHospital 实验记录

> 生成时间：2026-03-23  
> 评测场景：OldHospital（室外建筑，1077帧），room_0（室内），Stairs  
> 评测指标：val rot_err（中位数角度误差），< 1° 比例

---

## 核心结论

**最佳有效结果 (OldHospital)**：**exp188** → `rot=1.11°`  
方案：depth-warp + MSFlowPoseNet + SD/DINOv2多尺度特征 + AE特征压缩 + pose-warp增强

**最佳无效结果**：exp059/065/066 → `rot≈0.03-0.04°`（验证集=训练集的自动分割，overfitting信号）

**Transformer方案彻底失败**：exp070/071/072 → `rot≈5.7°`（位姿更新≈0，3DGS特征域差距太大）

---

## 实验历程

### 阶段一：基线建立（exp054-067）- MSFlowPoseNet + 3DGS渲染特征

| 实验 | 最佳val rot | 场景 | 关键改动 |
|------|------------|------|--------|
| exp054 | 3.32° | OldHospital | MSFlowPoseNet基线，PCA特征 |
| exp055 | 3.48° | OldHospital | 变体 |
| exp058 | 3.55° | OldHospital | 变体 |
| exp059 | **0.04°** ⚠️ | OldHospital | depth-warp首次引入（val=trainset子集，结果不可信） |
| exp060 | 2.97° | room_0 | 单尺度基线 |
| exp061 | 3.22° | OldHospital | TransformerPoseNet V1（LoFTR样式） |
| exp062 | 3.51° | OldHospital | 从exp059 warmstart |
| exp063 | 3.04° | OldHospital | 变体 |
| exp064 | 3.31° | OldHospital | 从exp059 warmstart |
| exp065 | **0.04°** ⚠️ | OldHospital | depth-warp + ref增强（val=trainset子集，同exp059问题） |
| exp066 | **0.03°** ⚠️ | OldHospital | depth-warp + 强ref增强（val=trainset子集，不可信） |
| **exp067** | **1.17°** ✓ | OldHospital | depth-warp + **pose-warp增强**（真实验证分割），首次有效突破 |

**结论**：exp059/065/066的极低误差系val集与train集重叠导致，exp067才是第一个可信的depth-warp结果（1.17°）

---

### 阶段二：扩展到多场景（exp110-143）

| 实验 | 最佳val rot | 场景 | 关键改动 |
|------|------------|------|--------|
| exp110 | 5.39° | Stairs | 基线，FoV修正 |
| **exp110b** | **0.26°** ✓ | Stairs | batch=12，低噪声微调 |
| exp112 | 4.70° | OldHospital | oi=5 |
| exp113 | 0.14° | room_0 | 高分辨率 |
| exp114/115 | 1.24-1.25° | Stairs | 低噪声微调窗口 |
| **exp116** | **0.10°** ✓ | room_0 | 超低噪声，极佳结果 |
| exp126 | 4.21° | OldHospital | 低噪课程 |
| exp127 | 3.36° | OldHospital | 课程调度 |
| exp131 | 0.21° | Stairs | 低噪微调 |
| **exp134** | **3.29°** | OldHospital | oi=10，成为后续warmstart基础 |
| **exp141** | **0.19°** ✓ | Stairs | 低噪微调，最终版 |
| **exp143** | **0.12°** ✓ | room_0 | 低噪微调，最终版 |

**结论**：MSFlowPoseNet在室内/简单场景效果好（room_0: 0.10°），但OldHospital室外场景始终卡在3°+

---

### 阶段三：架构改进探索（exp136-161）

| 实验 | 最佳val rot | 关键改动 | 结论 |
|------|------------|--------|------|
| exp136 | 3.49° | PE concat（从exp134 warmstart） | 无明显提升 |
| exp137 | 3.51° | 去除IRLS | 无效 |
| exp138 | 3.27° | 激进learning rate | 略有改善 |
| exp139 | 3.45° | 跳过coarse阶段 | 无效 |
| exp140 | 5.50° | 结合多改进 | 负优化 |
| exp142 | 3.24° | DINO所有尺度（from exp138） | 无显著提升 |
| exp144 | 3.29° | 用DINO替换SD | 同等水平 |
| **exp145** | **3.60°** | oi=20 | 无效，噪声反而更难 |
| exp146 | 3.60° | DINO从头训练 | 无额外收益 |
| exp149 | 3.27° | 可定位性loss | 无效 |
| exp150 | 3.41° | flow一致性loss | 无效 |
| exp151 | 3.39° | 增强solver | 无效 |
| exp152 | 3.45° | 联合阶段1 | 无效 |
| exp154 | 3.41° | solver oi=15 | 无效 |
| exp155 | 0.42° | OldHospital: selected PCA特征 | 好于基线但配置有问题 |
| exp160 | 3.51° | flowfeat特征 | 无效 |
| exp161 | 3.87° | flowfeat v2 | 更差 |

**结论**：在OldHospital上，3DGS渲染特征的域差距（self-sim≈0.2）是根本瓶颈，架构改进无法解决

---

### 阶段四：room_0 Transformer架构探索（exp170-175）

| 实验 | 最佳val rot | 关键改动 | 结论 |
|------|------------|--------|------|
| exp170 | 1.42° | room_0 GNC-GM鲁棒求解 | 比基线差 |
| exp171 | 1.39° | room_0 凸化上升 | 比基线差 |
| exp172 | 3.12° | room_0 鲁棒结合 | 更差 |
| exp173 | 1.45° | Transformer refiner | 比基线差 |
| exp174 | 1.39° | Transformer流解码器 | 比基线差 |
| exp175 | 1.43° | 完整Transformer | 比基线差 |

**结论**：room_0上各种Transformer架构均不如基线MSFlowPoseNet（0.12°）

---

### 阶段五：DA3特征方案（exp180-188）

| 实验 | 最佳val rot | 关键改动 | 结论 |
|------|------------|--------|------|
| exp180 | 3.88° | DA3特征直接用 | 比SD/DINO差 |
| exp181 | < | DA3 perscale | 极差（配置错误） |
| exp182 | 3.88° | DA3 unified | 无改善 |
| exp183 | 4.22° | DA3 unified v2 | 无改善 |
| exp184 | 3.93° | DA3 norm fix | 无改善 |
| exp185 | 4.58° | DA3 低分辨率测试 | 更差 |
| exp186 | 3.51° | AE压缩特征 + 2DGS | 略好 |
| exp187 | 4.20° | AE diversity | 无效 |
| **exp188** | **1.11°** ✓ | **depth-warp + SD/DINOv2 AE压缩** | **当前最佳！** |

**结论**：DA3特征在OldHospital场景下不如SD+DINOv2。depth-warp是关键，配合正确特征（AE压缩的SD+DINO）达到1.11°

---

### 阶段六：Transformer定位网络（exp070-072）—— 完全失败

| 实验 | 最佳val rot | 关键改动 | 结论 |
|------|------------|--------|------|
| exp070 | 5.78° | TransformerPoseNetV2，DA3特征，E<40 | 失败 |
| **exp071** | **5.66°** | 14.7M参数，优化版，E139 | 失败 |
| **exp072** | **5.76°** | 3M参数，batch=16，E295/120 | 失败（训练完毕） |

**失败根因分析**：
- 3DGS渲染特征自相似性（self-sim=0.17-0.18），DA3存储特征（self-sim=0.85）
- 域差距导致cosine相似性≈0.33，位姿更新梯度≈0
- Depth-warp可将cosine提升到0.73（用joint_oh_v3模型）
- Transformer架构本身没问题，是特征来源问题

---

### 阶段七（进行中）：3DGS特征重建 perscale_v3

- `da3_perscale_v3`：用cos_weight=2.0 + grad_accum=16 + warmstart重新训练per-scale DA3特征
- 目标：提升rendered特征的可辨别性（从self-sim≈0.21降低到更好水平）
- PID: 1154486，输出：`output/feature_3dgs/oldhospital_da3_perscale_v3/`

---

## 当前活跃资产

| 资产 | 路径 | 用途 |
|------|------|------|
| 最佳定位模型 | `output/exp188_oh_depth_warp/checkpoints/best.pth` | OldHospital 1.11° |
| 2DGS几何模型 | `output/2dgs_models/OldHospital/v7_depth/` | exp188所需 |
| 特征3DGS | `output/feature_3dgs/oldhospital_ae_div_perscale/` | exp188所需 |
| 特征数据 | `output/features_multiscale_compressed/OldHospital_indexed/` | exp188所需 |
| 联合训练模型 | `output/2dgs_joint/joint_oh_v3/` | 新架构探索 |
| DA3特征 | `output/features_da3/OldHospital_indexed/` | 当前实验 |
| DA3统一特征 | `output/features_da3_unified/OldHospital/` | exp071所需 |
| 进行中训练 | `output/feature_3dgs/oldhospital_da3_perscale_v3/` | v3特征重建 |

---

## 待尝试方向

1. **depth-warp + DA3特征**（joint_oh_v3作为几何模型）：把1.11°推进 1°以下
2. **perscale_v3完成后**重新训练定位网络
3. **PixLoc-style纯高斯牛顿基线**：无神经网络，直接用depth-warp特征的图像Jacobian
4. **sparse 2D-3D匹配 + PnP后处理**：对flow结果做RANSAC精化
