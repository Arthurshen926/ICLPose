# ICLPose Phase 2 执行计划 — CLI Agent 指令

> 生成时间：2026-03-15  
> 基于 Phase 1 实验(exp149-154)结果分析  
> **所有操作必须串行执行，上一步完成后再做下一步**

---

## 0. Phase 1 实验结论（背景）

### 实验结果总表

| 实验 | 变量 | Best rot | 基线 | 结论 |
|:---|:---|:---:|:---:|:---|
| exp142 (OH baseline) | dino_all_scales | **3.24°** | — | 仅4 epoch即达最优，后续过拟合 |
| exp149 OH localizability | +localizability_prior | 3.27° | 3.24° | 无显著改善 |
| exp150 OH flow_consistency | +flow_consistency_weight | 3.41° | 3.24° | 有害，放弃 |
| exp151 OH enhanced_solver | +learnable_temp+dir_conf+adaptive_damp | 3.39° | 3.24° | 无改善 |
| exp152 OH phase1_combined | 全部叠加 | 3.45° | 3.24° | 多变量耦合失败 |
| exp153 R0 phase1_combined | 全部叠加 | 0.22° | 0.12° | 100ep收敛但不及基线 |
| exp154 OH solver+oi15 | solver + 15 outer_iters | 3.41° | 3.24° | 无改善 |

### 关键发现

1. **OH过拟合严重**：所有OH实验最优结果都出现在前5-10 epoch，100 epoch后退化到4.9-6.0°
2. **Phase 1所有features对OH无效**：solver增强、localizability、flow consistency均未突破3.24°天花板
3. **OH瓶颈在匹配架构**：trans=1300-1800mm说明匹配在大视角差区域系统性失败，r=4局部correlation窗口不够
4. **Room0 phase1 combined(0.22°)不及baseline(0.12°)**：可能因为同时改了太多变量 + 分辨率/数据路径不同
5. **flow_consistency_loss有bug**：mid分支存在时loss被额外/2，导致有效权重不一致

### 已放弃的方向
- ❌ `flow_consistency_weight` — 实验证明有害
- ❌ `adaptive_damping` — 条件数估计不准确（用对角线max/min近似≠真实条件数）
- ❌ 多变量同时改动 — 严格单变量消融

---

## 1. 代码修复（先做，不跑实验）

### 1.1 修复 flow_consistency_loss averaging bug

**文件**: `scripts/train_ms_flow.py`，`flow_consistency_loss()` 函数

**问题**: 当 `flow_mid is not None` 时 `total /= 2.0`，但 `flow_mid is None` 时不除，导致有效loss权重因配置而异。

**修复方案**:
```python
# 改为：
n_terms = 1
if flow_mid is not None:
    ...
    total = total + diff_mf
    n_terms += 1
total = total / n_terms
```

虽然flow_consistency已被放弃，但修复bug避免未来误用。

### 1.2 移除 adaptive_damping 的错误条件数估计

**文件**: `modules/geometry_solver.py`

**问题**: `diag.max() / diag.min()` 不等于真实条件数（不是特征值）。

**修复**: 要么改用 `torch.linalg.cond()` 或直接移除自适应逻辑，使用固定damping。当前所有成功实验都未启用此feature，保持`adaptive_damping: false`即可。

### 1.3 localizability_head 加训练预热

**文件**: `scripts/train_ms_flow.py`

**问题**: flow预测初期很差时，localizability supervision的soft label全趋向0（鸡生蛋问题）。

**修复**: 在训练前 `phase1_epochs` 期间将 `localizability_weight` 设为0，之后线性从0增加到目标值（用5个epoch线性warmup）。

---

## 2. OH 过拟合诊断与修复（高优先级）

### 2.1 exp155: OH 基线复现 + early stopping

**目标**: 确认OH在短训练下的最优epoch窗口

```yaml
# configs/exp155_oh_baseline_earlystop.yaml
# 复制 exp142_oh_dino_all_scales.yaml 的 ALL 设置
# 唯一变化：epochs: 20, val每epoch都做
# warmstart: output/exp142_oh_dino_all_scales/checkpoints/best.pth 的权重基础上继续微调是否有意义
# 或者 from scratch 跑20 epoch
exp_name: exp155_oh_baseline_earlystop
# 完整复制exp142的model/renderer/data段
# training:
#   epochs: 20
#   val_interval: 1  # 每epoch验证
```

**执行**: `CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py --config configs/exp155_oh_baseline_earlystop.yaml`

**分析**: 绘制rot/trans vs epoch曲线，确认最优epoch窗口。如果确实在epoch 0-3就最优，说明warmstart权重已经很好，微调反而过拟合。

### 2.2 exp156: OH降低学习率 + cosine annealing

**目标**: 测试OH是否因lr过高导致过拟合

```yaml
# configs/exp156_oh_lowlr.yaml
# 基于exp142 config
# 变化：lr: 0.000005 (原来0.00005的1/10), epochs: 30
```

**关键**: OH训练集可能太小导致过拟合。降LR + 短epoch可能帮助。

---

## 3. B3 FDA后精修（零训练成本，立即可做）

### 3.1 在exp142/exp143 best checkpoint上测试FDA后精修

**目标**: 验证网络输出pose经FDA精修是否能进一步改善

**步骤**:
1. 用 `scripts/eval_iterative.py` 获取exp142/exp143的per-sample pose预测
2. 将预测pose作为FDA初始值，用 `modules/featuremetric.py` 做Gauss-Newton精修
3. 需要加载原始高维特征（SD 1280d + DINO 768d）

**实现**: 创建 `scripts/eval_with_fda.py`:
```python
"""
FDA post-refinement evaluation.
Usage:
  python scripts/eval_with_fda.py \
    --config configs/exp142_oh_dino_all_scales.yaml \
    --checkpoint output/exp142_oh_dino_all_scales/checkpoints/best.pth \
    --fda_iters 10 --fda_damping 0.01
"""
# 1. Load model, run inference to get predicted poses
# 2. For each sample, load original SD+DINO features 
# 3. Use FeaturemetricAligner.align() to refine pose
# 4. Report before/after metrics
```

**场景**: Room0 + OldHospital

**依据**: FDA在5°噪声下已验证中位旋转0.48° (< 模型的3.24°)。如果模型给出的初始pose比5°好(大部分<=5°)，FDA应该能进一步精修。

---

## 4. B1 Attention-based Coarse Matching（核心创新，本阶段重点）

### 4.1 实现 cross-attention coarse matching

**目标**: 替代 `global_correlation()` 中的暴力all-pairs内积，用cross-attention增强coarse matching

**修改文件**: `ic_models/ms_flow_pose_net.py`

**设计**:
```python
class CrossAttentionMatcher(nn.Module):
    """
    替代 global_correlation()
    
    输入: query features (B, C, Hq, Wq), reference features (B, C, Hr, Wr)
    输出: attention-based correlation volume (B, Hr*Wr, Hq, Wq)
    
    架构:
    1. Flatten spatial → tokens: (B, N, C)
    2. 2层 cross-attention (query attends to reference)  
    3. 输出 attention map 作为 correlation volume
    
    关键参数:
    - n_heads: 4
    - n_layers: 2  
    - dim: 64 (decode_dim)
    """
```

**注意事项**:
- coarse分辨率 7×10=70 tokens (Room0) 或 15×26=390 tokens (OH)
- attention矩阵 390×390 仅~600KB显存，完全可行
- 保持输出shape与`global_correlation()`一致，不改下游FlowRefinementHead
- 用config flag `attention_coarse: true/false` 控制

**测试**: 先在Room0验证不退化，再在OH测试是否突破天花板

### 4.2 exp157: Room0 attention coarse matching

```yaml
# configs/exp157_r0_attention_coarse.yaml
# 基于exp032 config (Room0最优基线配置)
# 唯一变化: model.attention_coarse: true
# warmstart from exp032 best (如果有) 或 from scratch
```

### 4.3 exp158: OH attention coarse matching

```yaml
# configs/exp158_oh_attention_coarse.yaml
# 基于exp142 config
# 唯一变化: model.attention_coarse: true
# warmstart from exp142 best
```

---

## 5. 干净的Room0单变量消融（补充缺失数据）

之前Room0只跑了combined(exp153)，没有单变量消融。需要确认每个feature对Room0的独立效果。

### 5.1 exp159: Room0 + learnable_temperature only

```yaml
# 基于exp143 config (Room0 baseline, 0.12°)
# 唯一变化: model.learnable_temperature: true
# warmstart from exp143 best
# epochs: 50 (Room0没有过拟合问题)
```

### 5.2 exp160: Room0 + localizability_prior only

```yaml
# 基于exp143 config
# 唯一变化: model.localizability_prior: true, loss.localizability_weight: 0.01
# warmstart from exp143 best  
# epochs: 50
```

### 5.3 exp161: Room0 + IRLS (irls_iters=2)

```yaml
# 基于exp143 config
# 唯一变化: model.irls_iters: 2
# warmstart from exp143 best
# epochs: 50
```

---

## 6. A1 任务驱动AE微调（如果B1无效则做）

### 条件触发
- 如果 exp157/158 (attention coarse) 在OH上改善 < 0.3°
- 说明匹配架构不是唯一瓶颈，特征质量也需要提升
- 此时启动A1

### 实现概要
1. 加载 `feature_compression/autoencoder.py` 的 `AutoencoderFlexible` encoder
2. 在训练时将encoder嵌入forward pass，替代固定的离线压缩特征
3. 分阶段：冻结encoder 5 epoch → 解冻后2层 → 全解冻
4. 加 distillation loss 约束微调后特征不偏离原始分布

---

## 执行顺序（严格按序）

```
Step 1: 代码修复 (1.1 + 1.2 + 1.3)
  ↓ commit: "fix: flow_consistency averaging, localizability warmup"
Step 2: exp155 + exp156 OH过拟合诊断 (并行，GPU 0+1)
  ↓ 等训练完成，分析过拟合模式
Step 3: B3 FDA后精修脚本 + eval (3.1)
  ↓ 不需要训练，直接在现有checkpoint上测试
Step 4: B1 attention coarse 代码实现 (4.1)
  ↓ commit: "feat: CrossAttentionMatcher for coarse stage"
Step 5: exp157 R0 + exp158 OH attention实验 (并行，GPU 2+3)
  ↓ 等训练完成
Step 6: 根据exp157/158结果：
  - 如果OH改善 > 0.5° → 推进B1深化（mid层也用attention）
  - 如果OH改善 < 0.3° → 启动A1 AE微调
Step 7: R0单变量消融 exp159-161 (并行，GPU 0+1+2)
  ↓ 补充Room0的ablation数据
```

### GPU分配建议

```
当前运行(可随时中止，已完成100 epoch):
  GPU 0-5: exp149-154 (已完成或接近完成)

Phase 2:
  GPU 0: exp155 (OH early stopping) → exp159 (R0 learnable_temp)
  GPU 1: exp156 (OH low lr) → exp160 (R0 localizability) 
  GPU 2: exp157 (R0 attention) → exp161 (R0 IRLS)
  GPU 3: exp158 (OH attention)
  GPU 4-5: 备用 / FDA eval
```

---

## 成功标准

| 实验 | 成功基准 | 说明 |
|:---|:---|:---|
| exp155/156 | 确认OH最优epoch窗口 | 诊断性实验 |
| FDA后精修 | OH rot < 2.5° 或 R0 rot < 0.10° | 零训练成本的改善 |
| exp157 (R0 attention) | rot ≤ 0.12° | 不退化即可 |
| exp158 (OH attention) | **rot < 2.8°** | 突破3.24°天花板 ≥ 0.4° |
| exp159-161 | 确认每个feature的独立贡献 | 消融数据 |

---

## 注意事项

1. **严格单变量**: 每个实验只改一个config参数，其余完全一样
2. **warmstart一致**: OH从exp142 best，R0从exp143 best
3. **不要长训**: OH容易过拟合，20-30 epoch足够；R0可50 epoch
4. **每次commit**: 代码修改后先commit再跑实验
5. **日志检查**: 每个实验跑完后检查 ★ best 行和最后5个 [Val E] 行
6. **先杀旧进程**: Phase 1实验如果还在运行，先kill释放GPU

---

## 配置模板

所有新实验的config都应基于对应场景的baseline config：
- Room0 baseline: `configs/exp143_room0_lownoise_ft.yaml` (如果不存在，用 `exp032_cosine_fiters8.yaml`)
- OH baseline: `configs/exp142_oh_dino_all_scales.yaml` (如果不存在，用对应OH config)
- 查看baseline config: `python3 -c "import torch; c=torch.load('output/expXXX/checkpoints/best.pth',map_location='cpu')['config']; import yaml; print(yaml.dump(c))"`
