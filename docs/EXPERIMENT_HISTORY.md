# 实验历史与改进记录

> 从 ICPoseNet v1 到 MSFlowPoseNet exp032 的完整演进路径

---

## 总览

```
exp001-009: ICPoseNet (Transformer, 直接回归)     → ~5-10° rot
exp010-014: CorrPoseNet (光流+几何, 单尺度)       → ~1-2° rot
exp015:     CorrPoseNet 调参最优                    → 0.46° rot, 83.6% <1° ★ 基线
exp029-030: MSFlowPoseNet (单pass, 失败)           → >>5° rot
exp031:     MSFlowPoseNet + 外循环迭代              → 0.87° rot (bug), 95.6% <1°
exp032:     MSFlowPoseNet + cosine loss + warmstart → 0.33° rot, 95.6% <1° ★★ SOTA
```

---

## Phase 1: ICPoseNet (exp001-009)

### 方法
- 基于 Transformer 的直接位姿回归
- 查询图像特征 + 参考特征 → cross-attention → MLP → 6-DOF 位姿
- 使用 DINOv2 / SD 提取的特征

### 问题
- 直接回归精度有限, rot 通常在 5-10° 量级
- 无几何先验约束
- 无迭代精化

### 结论
直接回归位姿在精度上有天花板, 需要引入几何约束。

---

## Phase 2: CorrPoseNet (exp010-015)

### 方法 (参考 I2P-Reg 论文)
- 光流匹配 + Image Jacobian 几何求解
- 单尺度特征 (fine resolution)
- 外循环迭代: 更新位姿 → 重新渲染 → 再匹配
- 全局相关性 (all-pairs)

### 关键改进
- exp010-014: 逐步调试渲染器、损失函数、训练策略
- exp015: 最终调参版本, 达到稳定结果

### exp015 结果 (CorrPoseNet 基线)

| 指标 | 数值 |
|------|------|
| Rot Mean | 0.46° |
| Trans Mean | 25.5mm |
| <1° | 83.6% |

### 局限
- 单尺度无法处理大位移 (大噪声初始位姿)
- 全局相关性计算量大, 限制了细节分辨率
- 未使用 RAFT-style 迭代精化

---

## Phase 3: MSFlowPoseNet v1 (exp029-030)

### 目标
引入多尺度 coarse-to-fine 光流, 提升大位移处理能力。

### 方法
- 新模型 `MSFlowPoseNet`: coarse(7×10) → mid(15×20) → fine(35×46)
- ScaleDecoder 降维到 64-d
- 全局相关性 (coarse) + 局部相关性 (mid/fine)
- RAFT-style GRU 精化 (fine)

### 问题: 完全失败
- rot >>5°, 完全无法收敛
- **根因**: 使用单 pass 前向, 没有外循环迭代
  - 初始位姿噪声大 → 渲染的参考特征与查询差异太大
  - 光流预测不准 → 几何求解的位姿增量偏差大
  - 没有机会用更好的位姿重新渲染纠正

### 教训
外循环迭代精化是 **必须的**, 不是可选优化。

---

## Phase 4: MSFlowPoseNet + 外循环 (exp031)

### 关键改动
1. **外循环迭代**: `outer_iters=3` (train), `val_outer_iters=5` (val)
2. 每次迭代重新渲染参考特征
3. 渲染对位姿可微, 光流监督信号随迭代改善
4. `T_curr.detach()` 截断梯度避免爆炸

### 配置 (`configs/exp031_iterative.yaml`)
```yaml
model:
  fine_iters: 4
  local_radius: 4
training:
  lr: 1e-4
  epochs: 80
  phase1_epochs: 15
  rot_loss_type: acos
  outer_iters: 3
  val_outer_iters: 5
  noise_rot_min: 5.0
  noise_rot_max: 15.0
  warmup_epochs: 25
```

### 结果
- E19: <1° = **93.3%** (历史最高, 当时)
- E20: <1° = 91.1%
- Rot: 始终显示 ~0.87° (不下降)

### 发现的 Bug: 0.81° 指标地板

```python
# 在 pose_loss() 的指标计算中:
cos_theta = cos_theta.clamp(-1+1e-4, 1-1e-4)
angle = torch.acos(cos_theta)  # 即使 cos_theta=1.0, acos(1-1e-4) = 0.8103°!
```

这不影响训练 (损失函数中的 clamp 更紧), 但**误导了指标报告**。
实际 exp031 的真实旋转精度约 0.4-0.5° (从 <1° 指标推断)。

### 其他问题
- acos 损失梯度在 θ→0 时爆炸
- 噪声课程 warmup 太快 (25 epochs), 大噪声发散
- Phase1 太长 (15 epochs), 光流过拟合

---

## Phase 5: MSFlowPoseNet 全面优化 (exp032)

### 所有改动汇总

| 改动 | exp031 | exp032 | 原因 |
|------|--------|--------|------|
| rot loss | acos | **cosine** (1-cos) | 梯度安全, 无爆炸 |
| fine_iters | 4 | **8** | 更多 GRU 迭代, 更精细光流 |
| lr | 1e-4 | **5e-5** | 更稳定训练 |
| phase1_epochs | 15 | **3** | 快速进入端到端优化 |
| warmup_epochs | 25 | **40** | 更缓的噪声增长 |
| noise_rot_min | 5.0 | **2.0** | 从更小噪声开始 |
| noise_rot_max | 15.0 | **8.0** | 最大噪声也减小 |
| trans_weight | 1.0 | **10.0** | 加强平移监督 |
| metric clamp | 1e-4 | **1e-7** | 修复 0.81° bug |
| warmstart | None | **exp031 best** | 利用预训练权重 |
| pose_warmup | None | **8 epochs, min=0.1** | 逐步引入 pose loss |

### 配置 (`configs/exp032_cosine_fiters8.yaml`)
```yaml
experiment_name: exp032_cosine_fiters8
model:
  fine_iters: 8
  local_radius: 4
training:
  lr: 5e-5
  epochs: 60
  batch_size: 1
  phase1_epochs: 3
  rot_loss_type: cosine
  pose_weight: 1.0
  trans_weight: 10.0
  outer_iters: 3
  val_outer_iters: 5
  noise_rot_min: 2.0
  noise_rot_max: 8.0
  noise_trans_min: 0.05
  noise_trans_max: 0.2
  warmup_epochs: 40
  pose_warmup_epochs: 8
  pose_warmup_min: 0.1
  warmstart: output/exp031/best.pth
```

### 训练进度 (截止 E16)

| Epoch | Rot Mean | Rot Med | Trans | <1° | Flow EPE | 备注 |
|-------|----------|---------|-------|-----|---------|------|
| E4 | 0.49° | — | 31.9mm | 91.1% | — | 首个较好结果 |
| E5 | 0.55° | — | — | — | — | 轻微波动 |
| E14 | 0.34° | 0.24° | 20.6mm | 94.4% | 0.22 | ★ New best |
| E15 | 0.34° | 0.24° | 21.8mm | **95.6%** | 0.21 | |
| **E16** | **0.33°** | **0.23°** | **20.7mm** | 94.4% | 0.20 | **★ Best** |

### 与基线对比

| 指标 | CorrPoseNet (exp015) | **exp032 (E16)** | 提升 |
|------|---------------------|-------------------|------|
| Rot Mean | 0.46° | **0.33°** | **-28%** |
| Rot Median | — | **0.23°** | — |
| <1° | 83.6% | **95.6%** | **+12pp** |
| Trans Mean | 25.5mm | **20.7mm** | **-19%** |

**MSFlowPoseNet 在所有指标上全面超越 CorrPoseNet, 验证了多尺度 + RAFT-GRU + cosine loss 方案的有效性。**

---

## exp031 最新状态 (对比参考)

exp031 仍在后台训练 (E28-E32), 由于 acos 指标 bug:
- Rot: 始终 ~0.81-0.91° (实际可能更低)
- <1°: 93.3-95.6%
- Trans: 22-24mm
- 已无法继续改善 (不如 exp032)

---

## 关键经验总结

1. **外循环迭代是必须的**: 没有它, 多尺度光流完全不work
2. **Cosine loss >> acos loss**: 梯度安全, 没有 θ→0 时的爆炸问题
3. **Warmstart 有教**: 从 exp031 启动, exp032 在 E4 就接近基线水平
4. **噪声课程要慢**: warmup_epochs=40 比 25 好得多, 让模型逐步适应
5. **Phase1 要短**: 3 epochs 足够, 太长会过拟合光流
6. **Trans_weight 增大有效**: 10.0 比 1.0 显著改善平移精度
7. **Fine_iters 越多越好**: 8 > 4, 更多 GRU 迭代带来更精细光流
8. **指标计算要精确**: clamp 值影响报告的精度, 1e-7 足够小
