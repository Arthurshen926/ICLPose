# 2026-05-11 CPR selector / metric alignment update

## 新专家意见后的主结论

本轮按 `ChatGPT-特征训练与可微检索3.md` 的建议，重点验证：

1. score-map selector 是否真正消费 score map；
2. Phase6-A metric alignment 是否能让 K64 内 fine evidence 可排序；
3. 如果 feature evidence 不可排序，RGB / geometry evidence 是否能快速接上。

当前结论：

```text
K64/topK 内有好候选，但当前 query-map fine metric、RGB L1、coarse logit 都不能泛化识别 oracle。
delta prior 有 basin 信息，但只能到约 255mm，远达不到 0.25m/5deg < 100mm 的目标。
```

因此继续堆 frozen selector / score-map head / projector / query gate 不合理。

## 关键实验结果

### A. K64 cache evidence diagnostics, 0.25m / 5deg

同一 random K64 val cache，oracle 约 `92.0mm`。

| Evidence | pred mean | Spearman | good/bad AUC | 结论 |
|---|---:|---:|---:|---|
| delta_min | `255.3mm` | `0.673` | `0.869` | 只能保 basin，不能吃 oracle |
| score_mean, A2 | `401.2mm` | `0.051` | `0.548` | 近随机 |
| score_topk_mean, A2 | `376.8mm` | `0.038` | `0.523` | 近随机 |
| score_mean, Phase6-A shared | `390.9mm` | `0.052` | `0.548` | 未改善 |
| score_mean, A4 query gate | `409.0mm` | `0.046` | `0.549` | 未改善 |
| RGB L1 | `415.6mm` | `-0.002` | `0.483` | 不可用 |

score-map selector `expects_score_map=true`，cache 含 `[N,64,3,68,120]` score map；shuffle ablation 已接入。当前失败不是“score map 没接上”。

### B. Query/projector metric alignment

Phase6-A 小扰动 corr 指标可改善，但迁移不到 25cm K64 rerank：

```text
small-perturb corr EPE / peak acc 改善
25cm K64 score_mean Spearman 仍约 0.05
25cm K64 score_mean pred 仍约 390-409mm
```

A4 query-channel gate 只把 candidate-render-score val 从约 `252.9mm` 推到约 `250.3mm`，不是突破。

### C. Geometry / scene-coordinate diagnostic

新增 scene-coordinate head 方向作为 geometry evidence 试探。

| Variant | data | trainable | best val scene err |
|---|---|---|---:|
| head-only quick | 192 train / 32 val | scene head | `692cm` |
| head+warp quick | 192 train / 32 val | scene head | `716cm` |
| head-only full | 895 train / 64 val | scene head | `914cm` |
| fine-head full | 895 train / 64 val | scene head + fine head | `960cm` |
| stage4 full | 895 train / 64 val | scene head + fine/stage4 | `965cm` before stopped |

结论：当前轻量 scene-coordinate head 只能学到米级甚至十米级，不能直接作为 CPR 主证据。

## 本轮代码改动

- `train_fine_selector_cache.py`
  - 增加 `--eval-only`，可直接诊断 cache evidence，不再为了看指标误训 selector。
  - 增加 rank diagnostics：oracle、candidate0、selector Spearman/AUC、多种 prior/evidence baseline。
  - 增加 RGB evidence diagnostics：`rgb_l1_mean` / `rgb_l1_zscore` / `rgb_valid_fraction`。
  - 修正 constant-score Spearman，避免均匀 logits 被候选顺序伪相关污染。
- `train_impl.py`
  - candidate render score 支持 soft pose-error target。
  - projector / projected feature health 相关配置已补齐。
- configs
  - A3/A4 metric alignment / query-gate configs。
  - cached score-map RGB/query-gate diagnostics。
  - Phase6-B scene-coordinate diagnostics。

验证：

```text
python -m py_compile ... 通过
pytest tests/test_adaptive_joint_query_map.py -k ... 8 passed
```

## 调整后的主线建议

短期停止：

```text
frozen K64 selector
继续调 score-map head / query gate / local-corr projector
RGB L1 rerank
轻量 scene-coordinate head
WLS step size / map candidate-gradient / coarse adapter
```

当前瓶颈应改写为：

```text
已有 topK coverage，但现有证据不能泛化地区分 topK 内 5-25cm 级 pose 差异。
```

下一步如果继续冲 CPR 目标，应该换更强证据形式，而不是继续调 selector：

1. 引入显式 2D-3D correspondence supervision：用深度/可见点生成 sparse or dense query-to-map correspondences，训练 matcher/confidence，而不是只用 RADIO/DCFF local corr。
2. 用可解释的 PnP/RANSAC 或 differentiable PnP 作为 fine stage，selector 只做 failure/confidence gating。
3. 如果没有可靠 2D-3D supervision，当前 CPR learned selector claim 应降级，主论文不要声称 learned topK selector 已解决。

