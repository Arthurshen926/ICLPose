# ICLPose

**User:** Anonymous  
**Created:** 5/16/2026 12:54:37  
**Updated:** 5/24/2026 17:40:28  
**Exported:** 5/24/2026 17:43:27  
**Link:** [https://chatgpt.com/c/6a07f88d-362c-83ea-a904-b9f42272c2e9](https://chatgpt.com/c/6a07f88d-362c-83ea-a904-b9f42272c2e9)  

## Prompt:
2026/5/24 16:04:53

又朝着我们讨论的方向推进了一轮，git仓库radio分支已更新，请分析一下目前的进展和主要问题是什么

## Response:
2026/5/24 16:04:54

我会先核对最新 docs、localizability 代码和结果表，再从已成立证据、未成立 claim、主要 blocker 与下一步优先级四方面判断。

## Response:
2026/5/24 16:04:57

> **浏览GitHub仓库文档并分析进展**
>
> [ICLPose/docs/superpowers/plans at radio · Arthurshen926/ICLPose · GitHub](https://github.com/Arthurshen926/ICLPose/tree/radio/docs/superpowers/plans)
>
> **查找和查看最新的计划文件**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md)
>
> **查找最新的行并打开54行之后**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md)
>
> **点击查看文件内容及相关信息**
>
>
> **查看 GitHub 文件或原始内容**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_extract/localizability/selector.py)
>
> **分析代码请求并提供评估**
>
> 需要提供答案，可能需要提到最新的代码有格式问题但内容正常。也许需要检查README文件？也许可以接受。需要在答案中引用所有相关来源，特别是repo中的事实。我还需要总结当前进展和主要问题，用户提到git已更新并且文档是最新的，因此需要深思熟虑，不仅仅是重复信息，还需给出评估。
>
> **总结项目进展和瓶颈**
>
> 进展如下：
> - 主要功能定义已优化，如POFD-FS特性集成和假设本地化；架构边界已实现。
> - 模块在：选择器、评分方法、损失函数等方面已经完成，但仍有待稳定。
> - 主要问题在于“特征选择”并未完全得到验证，pose匹配器有改进，但显式选择器未得到充分因果证明。
>
> 已思考 4m 26s

## Response:
2026/5/24 16:09:21

你的最新进展**不是完全无效**，但目前支持的 claim 仍然很窄。准确判断是：

> **项目已经从混乱的 CPR / pose refinement 分支，收敛到一个更合理的 POFD-FS 主线：从 frozen foundation features 中学习适合先验地图定位假设判别的 compact feature subspace。**  
> 但目前证据只支持 **OldHospital controlled candidate ranking 上的初步有效性**，还不支持 **通用先验地图定位特征选择方法**，更不支持 **TRO 级完整定位系统或真实部署提升**。

下面分层讲。

---

# 1. 当前最重要的正进展

## 1.1 主线比之前清楚了

现在文档已经把项目定位为 **feature-centric localization paper**，核心问题是：

```text
Which foundation features localize?
```

具体任务是：

```text
score(query, map/reference, candidate hypothesis)
```

而不是继续把 Stage4 continuous CPR、LoFTR refinement、WLS/GN、feature-metric update 当主线。文档也明确说：只有 POFD-FS 自己在 real-init final pose 上有提升时，才可以 claim SOTA continuous camera pose refinement；否则这些分支只是 diagnostic / external baseline。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

这是对的。以前最大的问题是你同时想做：

```text
foundation feature selection
3D feature reconstruction
candidate reranking
continuous pose refinement
solver handoff
SOTA localization
```

现在至少主线压缩成：

```text
frozen foundation feature
→ localization feature selector / pose-adapted feature
→ compact feature + utility / uncertainty
→ hypothesis scorer
→ pose-distance / basin-aware ranking
```

这个方向比之前优雅得多。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

---

## 1.2 工程边界明显更清楚

现在 `feature_extract/localizability/` 下面已经有一套独立边界：

```text
selector.py
scorer.py
losses.py
failure_replay.py
handoff_cache.py
reference_pose_bank.py
candidate_bank.py
metrics.py
mapability.py
visualization.py
```

文档也把这些作为当前 POFD-FS 实现边界。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

代码层面，`LocalizationFeatureSelector` 已经不是一个空概念，而是实现了：

```text
grouped channel gate
1x1 compact projection
spatial utility head
uncertainty head
L2-normalized selected feature
```

它的目标是从 frozen dense features 中选择 compact localization-usable subspace。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_extract/localizability/selector.py))

`PoseHypothesisScorer` 支持：

```text
same_pixel
local_corr
pair_matcher_local
```

并且 pair-matcher 路径能结合 `query_utility`、render valid mask 和局部 offset scoring。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_extract/localizability/scorer.py))

loss 也已经比较贴合 POFD-FS 主线，包括：

```text
pose-distance soft listwise KL
basin BCE
online hard negative loss
channel sparsity
spatial utility entropy
```

这些都比之前“teacher reconstruction / confidence / Fisher 到处试”的状态更集中。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_extract/localizability/losses.py))

---

## 1.3 OldHospital controlled ranking 有明确正结果

目前最强证据仍然来自 OldHospital controlled val128：

| feature / scorer | q10 pred | q25 pred | q50 pred | q50 top1 | q50 gap | q50 Spearman |
|---|---:|---:|---:|---:|---:|---:|
| raw RADIO + local r4 | 0.086 | 0.189 | 0.300 | 0.508 | 0.171 | 0.664 |
| query student + local r4 | 0.059 | 0.140 | 0.274 | 0.594 | 0.144 | 0.714 |
| pose-adapted + pair matcher r16 | 0.045 | 0.107 | 0.223 | 0.719 | 0.093 | 0.586 |

这个表支持两个重要结论：

1. **raw RADIO dense feature 不是最适合 localization hypothesis ranking 的特征。**
2. **经过 query-student / pose-adapted / pair-matcher-local scoring 后，定位候选排序确实改善。**

尤其 q10/q25 改善比较强；q50 也有明显提升，但还没过目标阈值。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

---

## 1.4 q50 失败机制被定位得更清楚了

现在 q50 不再是“感觉不好”，而是已经有比较清楚的 failure diagnosis。

当前 q50 target 是：

```text
pred <= 0.21
top1 >= 0.75
gap <= 0.09
Spearman >= 0.55
```

当前主结果是：

```text
pred 0.223
top1 0.719
gap 0.093
Spearman 0.586
```

小 calibrator 最好能到：

```text
pred 0.2129
top1 0.7422
gap 0.0835
Spearman 0.5911
```

但它仍然没有稳定过 promotion threshold。多 seed 结果也显示它只能把 q50 从 `0.2227m` 稳定推到大概 `0.213-0.221m`，但不能可靠超过 `pred <= 0.21` 和 `top1 >= 0.75`。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

更关键的是，现在已经知道 q50 的主要失败不是简单容量不够，而是 **train / val candidate distribution mismatch**：

| split | samples | wrong top1 | near-identity wrong | outside-basin wrong | wrong selected mean | wrong oracle mean |
|---|---:|---:|---:|---:|---:|---:|
| train q50 | 96 | 55 | 5 | 29 | 0.333m | 0.204m |
| val q50 | 128 | 36 | 26 | 33 | 0.461m | 0.129m |

也就是说，val q50 的失败大量是 **near-identity false positives**，但 train q50 没有充分暴露这种失败分布。文档也明确结论：剩余 q50 gap 是 candidate-bank coverage problem，不是继续加大 calibrator 的问题。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

这是一条很有价值的结论。

---

## 1.5 public reference-pose benchmark 基础已经搭起来

你现在已经构建了 Cambridge 五场景的 reference-pose candidate bank：

| scene | queries | top1 median trans | oracle@10 median trans | note |
|---|---:|---:|---:|---|
| GreatCourt | 760 | 6.724m | 2.869m | large rerank headroom |
| KingsCollege | 343 | 2.909m | 1.985m | moderate rerank headroom |
| OldHospital | 182 | 3.978m | 2.276m | large rerank headroom |
| ShopFacade | 103 | 1.377m | 0.736m | usable second scene |
| StMarysChurch | 530 | 2.839m | 1.458m | large rerank headroom |

这不是 POFD-FS positive result，但它是非常重要的 public benchmark substrate。它让项目终于有机会脱离 OldHospital-only 证据。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

---

## 1.6 Mapability 和 interpretability 也从“概念”变成“接口”

mapability diagnostic 已经实现，可以计算 projected track feature variance。当前 raw RADIO baseline 是：

| feature | images | observations | tracks | dim | variance |
|---|---:|---:|---:|---:|---:|
| coarse_sem | 40 | 1236 | 567 | 64 | 1.4404 |
| fine_geo | 24 | 163 | 59 | 64 | 0.0033 |

文档也明确说这只是 raw RADIO baseline，还没有证明 selected POFD-FS mapability。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

interpretability 方面，counterfactual diagnostics 已经实现，可以做：

```text
channel_group_counterfactual_drop
spatial_utility_counterfactual_drop
```

测试也验证了移除高 utility channel / spatial evidence 会比移除低 utility evidence 更影响 selected pose cost。但这还需要 exported selected-feature score maps 和 utility maps 才能成为 paper result。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

这说明工具链正在成型，但还没变成论文证据。

---

# 2. 当前最主要的问题

## 2.1 目前只支持“controlled hypothesis ranking”，不支持完整定位系统

这是最核心问题。

当前 `pred_cost_m` 是：

```text
top1 ranked candidate 的 pose error
```

不是：

```text
refined final pose error
```

当前 `basin_recall@K` 也只是几何 proxy：

```text
topK 是否包含低于某个 translation / rotation threshold 的 candidate
```

它不是 solver-measured basin。文档已经明确：final localization claim 需要 solver-specific handoff protocol。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

所以目前能 claim 的是：

> POFD-FS improves solver-free hypothesis-localizability on OldHospital controlled banks.

不能 claim：

> POFD-FS improves final localization accuracy.

更不能 claim：

> POFD-FS is a SOTA map-based relocalization system.

---

## 2.2 Handoff solver 目前是负结果

你们已经实现了 handoff evaluator 和 init cache export，但目前结果说明：现有 solver handoff 并不能把 POFD-FS ranking 转化为 final pose improvement。

q50 val128 上：

| init cache | init mean | init median | refined mean | refined median | conclusion |
|---|---:|---:|---:|---:|---|
| POFD top1 | 217.6mm | 125.0mm | 219.0mm | 146.0mm | no improvement |
| POFD top4, PNP-inlier choice | 294.6mm | 125.0mm | 218.7mm | 146.0mm | mean improves from bad init, still worse than POFD top1 |
| POFD top8 oracle diagnostic | 129.4mm | 125.0mm | 207.1mm | 143.7mm | solver degrades even oracle candidates |
| POFD top1, 2 refinement iterations | 217.6mm | 125.0mm | 221.2mm | 148.7mm | second iteration does not help |

结论非常明确：**当前 final localization result 应该仍然是 selected candidate pose = POFD top1；naive render-LoFTR refinement 不是可靠 handoff solver。** ([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

这意味着，POFD-FS 目前不能靠 handoff solver 来增强 claim。后续如果要做 solver-conditioned localization，必须换成真正独立且稳定的 solver，例如文档提到的 GS-SMC-style RGB rendered-view refinement，但必须在同一 exported-cache protocol 下打赢 top1 identity handoff。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

---

## 2.3 q50 剩余问题不是 loss 小修，而是 candidate-bank 设计问题

你们尝试了几个方向：

### failure replay

直接用 cached selected-wrong vs oracle margin loss：

```text
failure-only, pair loss w2, lr5e-5 → no improvement
all-train, pair loss w4, lr2e-5 → no improvement
```

文档结论：direct backprop into heavy pair-matcher score path memory-heavy、batch size 1，而且没移动 validation selection。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

### small score calibrator

能改善一点，但不稳定过线。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

### generic near-init append

添加 identity + jitter candidates 后，训练集分布变了，但 val 没改善；甚至 top25 near-init train calibrator 比原 top16 train 更差。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

### mined replay

当前 train hard pairs 仍然无法复现 val near-identity trap。val top16 里 `107/128` mined negatives 是 near-identity，但 train top16 和 top25 near-init 都只有 `4/96`。静态 replay 也没过线。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

因此当前 q50 最重要结论是：

> **不能继续靠 scalar calibrator、generic near-init append、static replay。必须从更大的 randomized/jittered candidate pool 中，用 frozen scorer 主动挖 score-high wrong candidates，再构建新的 score-hard training bank。**

这点非常关键。否则你会继续在不覆盖真实 failure mode 的 train table 上调参。

---

## 2.4 public multi-scene 目前还是负结果 / 未完成

reference-pose bank 搭好了，但 raw RADIO pooled-feature reranking 和 query-student pooled reranking 都是负结果。

OldHospital / ShopFacade 上，raw RADIO pooled score 都比 retrieval-order baseline 差：

| method | OldHospital pred | ShopFacade pred |
|---|---:|---:|
| retrieval order | 4.186m | 1.649m |
| raw RADIO fine pooled | 5.147m | 2.254m |
| raw RADIO coarse pooled | 5.187m | 2.048m |

query-student fine pooled 也不行，OldHospital top10 reference-pose ranking 上：

| Method | pred | oracle | gap | top1 | Spearman | basin@1 | basin@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Retrieval order | 4.186 | 2.646 | 1.540 | 0.181 | 0.316 | 0.655 | 0.897 |
| Raw RADIO fine pooled | 5.147 | 2.646 | 2.500 | 0.137 | 0.051 | 0.494 | 0.770 |
| Raw RADIO coarse pooled | 5.187 | 2.646 | 2.541 | 0.088 | 0.127 | 0.425 | 0.851 |
| Query-student fine pooled | 5.195 | 2.646 | 2.548 | 0.093 | 0.065 | 0.345 | 0.690 |

结论是：**global pooled raw/query-student foundation features 不适合 public reference-pose reranking。** ([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

这对论文是双刃剑：

- 它支持 “raw foundation feature similarity is not enough”。
- 但它也说明你还没有 public multi-scene positive result。

文档也明确说，当前 available feature caches 只覆盖 OldHospital / ShopFacade 的 raw RADIO，KingsCollege / GreatCourt / StMarysChurch 还没有；query-adaptive cache 也只有 OldHospital debug export。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

所以，现在不能 claim：

```text
POFD-FS works across Cambridge scenes.
```

---

## 2.5 当前“特征选择”因果性还没有真正证明

虽然 `LocalizationFeatureSelector` 已经实现了 grouped channel gate、projection、utility、uncertainty，但当前最强结果实际来自：

```text
pose-adapted + pair matcher r16
```

而不是一个已经被充分证明的 explicit selector。文档里也记录过 naive selector / projection 训练没有明显改善，当前正结果主要依赖 rendered-candidate + low-res large-radius pair-matcher scoring。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

现在 interpretability 接口有了，但还没有 paper-level 结果：

```text
selected-feature score maps
utility maps
channel-group counterfactual table
spatial top/bottom masking table
```

因此当前还不能强 claim：

> 我们明确选择出了 foundation feature 中哪些 channel / regions 对定位有用。

更准确的 claim 仍然是：

> pose-adapted foundation feature path improves controlled hypothesis ranking.

这比“feature selection”弱一档。

---

## 2.6 Mapability 仍然没有闭环

本项目如果要叫 **foundation feature selection for prior-map visual localization**，必须证明 selected feature 能进入 3D map。

目前只有 raw RADIO projected-track variance baseline。文档明确说：

> It does not yet prove selected POFD-FS mapability because full selected/query-adaptive dense caches are not exported. ([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

所以当前还没有完成：

```text
selected feature
→ multi-view aggregation
→ selected 3D feature map
→ rendered selected feature
→ same hypothesis scoring pipeline
```

没有这个闭环，论文会被质疑只是一个 2D / cached candidate scorer，而不是 map-based localization feature.

---

## 2.7 工程可靠性仍然有隐患

一个例子是 near-init cache probe 中发现：

> existing eval config silently truncated candidate caches to top16; top25 config was required to actually score appended candidates. ([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

这类问题非常危险，因为它会让实验结果看起来跑了，但实际上没有评估到新增候选。

另一个例子是 descriptor export 修复了历史 teacher cache 依赖、student export resolution、legacy `fine_loc_*` checkpoint keys 等问题。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

这说明工程已经在变好，但仍需要更严格的 regression checks，尤其是：

```text
candidate count sanity
feature cache shape sanity
score source sanity
external PnP leakage check
eval split sanity
```

---

# 3. 目前得到的主要结论

我会把当前结论分成“可靠结论”和“暂时不能下的结论”。

## 3.1 可靠结论

### 结论 A：raw foundation feature similarity 不足以做定位

public reference-pose reranking 中，raw RADIO pooled descriptors 和 query-student pooled descriptors 都比 retrieval-order baseline 差。这说明简单把 foundation feature 做 global pooling / cosine similarity，并不能变成一个有效的 localization scorer。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

### 结论 B：controlled rendered-candidate ranking 是当前最有效路径

OldHospital q10/q25/q50 上，positive evidence 仍然来自：

```text
rendered candidate
low-res large-radius pair-matcher scoring
pose-adapted feature
```

而不是 public pooled reference-pose scorer。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

### 结论 C：q50 难点是 score surface + hard negative distribution

q50 val failures 是 near-identity score traps，train 没覆盖；generic near-init append 没有生成真正 score-hard wrong candidates；static replay 也不够。因此下一步必须做 **large pool → frozen scorer mining → score-hard training bank**。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

### 结论 D：naive solver handoff 不可靠

PNP-quality fields、render-LoFTR handoff、2-iteration refinement都没带来最终定位提升；甚至 oracle top8 candidates 也被 render-LoFTR refinement 降级。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

### 结论 E：当前 POFD-FS 是 solver-free localizability evidence，不是 final localization system

这是当前最重要的论文边界。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md))

---

## 3.2 目前不能下的结论

不能说：

```text
1. POFD-FS 已经是完整先验地图定位方法。
2. 选出来的 feature 已经可稳定重建成 3D feature map。
3. POFD-FS 在 public multi-scene 上有效。
4. selected feature 可以替代 LoFTR / MASt3R 做 sparse correspondence。
5. q50 已稳定达标。
6. 下游 solver handoff 已经成立。
7. 当前显式 selector 已经因果证明了“哪些 feature 被选择且有用”。
```

这些都还缺证据。

---

# 4. 我对当前项目状态的判断

我会给出一个比较明确的定性：

> **当前项目已经从“混乱探索”进入了“可诊断主线原型”阶段。它不再是完全跑偏，但仍未达到论文主方法成立的状态。**

更具体：

```text
方向：比之前更正确。
工程：边界清楚很多，但仍需清理和回归。
结果：OldHospital controlled ranking 有正证据。
q50：接近但没稳定过线。
public：有 benchmark substrate，但正结果未成立。
mapability：有工具，无 selected-feature 证据。
handoff：当前负结果。
claim：只能 claim solver-free hypothesis-localizability 的初步有效性。
```

---

# 5. 下一步最关键的优先级

我建议不要再“同时推进很多线”。按优先级应该这样走。

## 第一优先级：q50 score-hard candidate bank

当前最该做的不是继续 calibrator、不是 static replay、不是 generic near-init。

应该做：

```text
1. 生成大规模 randomized / jittered q50 candidate pool。
2. 用 frozen current scorer 评分。
3. 对每个 query 挖：
   positive = lowest-cost candidate
   negative = highest-score candidate with cost >= positive + gap
4. 强制 negative 包含 near-identity false positives 和 repeated-structure false positives。
5. 构建新的 q50 train bank。
6. 重新训练 hypothesis scorer / selector。
```

目标不是只提高 val，而是让 train / val failure distribution 对齐。

---

## 第二优先级：public reference-pose 不要用 global pooling

当前 pooled feature route 已经负了。下一步要做 patch-level / local-evidence scorer：

```text
query patches ↔ reference patches
top-M local evidence aggregation
utility-weighted similarity
hard-negative rank loss
```

候选仍然用 Cambridge top10 reference-pose banks。目标是至少在 OldHospital / ShopFacade / KingsCollege 中拿到正结果。

---

## 第三优先级：selected feature mapability export

不要先导出 full dense cache。先做 track-observation descriptor export：

```text
for each track observation:
    sample selected feature z(u)
    sample utility m(u)
    sample uncertainty σ(u)
```

然后比较：

```text
raw RADIO
query-student
POFD-FS selected
```

指标：

```text
track variance
primitive variance
rendered ranking retention
storage dim
```

这一步决定项目能不能从 “2D scorer” 升级为 “prior-map localization feature”。

---

## 第四优先级：interpretability 成为 paper evidence

需要真正跑出：

```text
channel group removal table
spatial high/low utility masking table
qualitative utility maps
failure-case visualization
```

目标是证明：

```text
selected high-utility evidence 对候选排序有因果贡献。
```

---

## 第五优先级：handoff solver 暂时降级

在新的 stable q50 bank 和 mapability 结果之前，不要继续强推 solver handoff。

当前结果已经说明：

```text
render-LoFTR+PnP handoff 是负结果；
PNP-quality fields 不可靠；
top1 identity 反而是当前最稳 final pose。
```

后续如果要做，应该以 GS-SMC-style fixed solver 为独立 external solver，并且必须打赢 top1 identity exported-cache protocol。

---

# 6. 最终结论

目前进展可以总结为：

> **POFD-FS 主线已经变得更清楚，工具链也更完整；OldHospital controlled hypothesis ranking 上有真实正结果，尤其 q10/q25 明确有效，q50 接近但未稳定过线。当前最大的技术问题是 q50 score surface 被 near-identity false positives 困住，而训练候选分布没有覆盖这种失败模式。当前最大的论文问题是 public multi-scene、selected-feature mapability、interpretability causal evidence 和 solver-conditioned downstream utility 仍然缺失。**

所以，当前项目**不是没有希望**，但距离“可以支撑强 claim”还差四个关键证据：

```text
1. q50 robust score-hard training 过线；
2. public reference-pose / second-scene 正结果；
3. selected feature mapability 正结果；
4. feature selection 的因果解释性结果。
```

在这四个证据出来之前，最合理的 claim 仍然是：

> **POFD-FS provides preliminary evidence that pose-adapted foundation features can improve solver-free map-conditioned hypothesis ranking in controlled settings.**

还不能 claim：

> **POFD-FS solves prior-map localization hard cases as a complete deployable localization method.**

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md)

## Prompt:
2026/5/24 16:53:13

不是为什么还用coarse_sem和fine_geo，这两个不是之前提取的radio中间层特征吗？为什么不用单独的radio原始特征？现在的精度怎么才是米级？这肯定不符合任务需求啊
请抛开目前已有的所有项目实现，从头到尾全新设计一下本项目方案，并且用同领域常用的词汇和表达来梳理方法，越清晰直观越好

## Response:
2026/5/24 16:53:17

我会把现有实现完全放下，重新按 prior-map visual localization 的标准流程设计：任务定义、地图表示、候选生成、特征学习、候选验证、可选精修和评估协议。重点先纠正 coarse_sem/fine_geo 与米级结果暴露出的任务错配，把“选特征”从历史命名和旧代码假设里解放出来。

## Response:
2026/5/24 16:53:31

此外，米级 reference-pose reranking 只能视为粗定位诊断；真正方案必须以最终 6DoF 闭环为目标，分清候选召回、几何验证和位姿求解/精修。

## Response:
2026/5/24 16:58:55

你这个质疑是对的。**如果现在还在用 `coarse_sem / fine_geo` 这类历史缓存名，并且候选排序结果还是米级，那么它已经偏离“基于先验地图的精确定位”任务了。** 米级结果最多说明“粗地点识别 / reference-pose retrieval”还有信号，不能支撑“定位特征选择”的强 claim，更不能支撑 TRO 级 prior-map visual localization。

我建议彻底重置设计，不再沿用现在仓库里的 coarse/fine、Stage2/Stage3/Stage4、candidate cache 等历史概念。新的项目应该从下面这个核心问题出发：

> **如何从 frozen vision foundation model 的原始 dense tokens 中，选择一个紧凑、可解释、可建图的 localizable subspace，用于先验地图条件下的困难定位假设验证，并最终提升下游定位成功率。**

这个问题比“用 RADIO 做 refinement”更清楚，也比“做一个候选排序器”更严谨。

---

# 1. 先明确：不要再用 `coarse_sem / fine_geo` 作为方法概念

`coarse_sem` 和 `fine_geo` 如果只是历史中间层缓存名，就不应该出现在新方法论里。它们最大的问题是：

```text
1. 名字暗示它们分别是语义和几何，但这不一定被证明；
2. 它们来自旧 pipeline 的中间层，不是一个干净的 foundation feature input；
3. 它们会让审稿人觉得方法是工程拼接，而不是从原始基础特征中学习选择；
4. 它们无法回答“到底选了哪些 foundation feature、为什么选、选了之后是否可建图”。
```

新的设计里应该只使用这个术语：

```text
raw VFM token bank
```

也就是从 frozen foundation model 中直接提取的原始 dense features，例如：

```text
RADIO / C-RADIO spatial tokens
DINOv2 spatial tokens
可选的 SigLIP / CLIP dense tokens
不同 layer / resolution 的 token bank
```

RADIO / AM-RADIO 的合理性在于它本来就是多教师蒸馏，把 CLIP、DINOv2、SAM 等视觉基础模型的不同能力融合到一个统一 backbone 中；DINOv2 的合理性在于它被设计为 all-purpose visual features，并在 image-level 和 pixel-level 任务中表现很强。([arXiv](https://arxiv.org/abs/2312.06709?utm_source=chatgpt.com))

所以新方法不应该说：

```text
我们使用 coarse_sem 和 fine_geo。
```

而应该说：

```text
我们从 frozen VFM token bank 中学习一个 localizable feature subspace。
```

如果最终发现某些 RADIO 中间层最有用，那也应该通过 layer/channel utility analysis 证明，而不是先验命名为 semantic / geometric。

---

# 2. 当前米级精度为什么不符合任务需求

米级结果通常说明你在做的是：

```text
reference-pose retrieval
map-cell retrieval
coarse place recognition
```

而不是精确 6DoF localization。

如果 candidate hypothesis 是 reference image pose，那么 top1 候选天然可能离 query 几米远。它只能作为 coarse prior，不能作为最终定位结果。已有 visual localization 文献也明确指出，image retrieval 在定位里常用于两件事：提供近似 pose，或者确定 query 可能看到的地图区域；但传统 retrieval performance 并不总能预测 localization performance，因此需要面向定位范式设计检索 / 候选评分。([arXiv](https://arxiv.org/abs/2011.11946?utm_source=chatgpt.com))

所以新项目必须把输出层级分清：

```text
Place-level output:
    米级可以接受，用于候选区域召回。

Hypothesis-verification output:
    应该把 top-K 候选压到 solver basin 内，例如 <0.5m / 10° 或 <0.25m / 5°。

Final localization output:
    室内/机器人定位至少应追求厘米到分米级，例如 <10cm / 5° 或 <25cm / 10° success。
```

如果当前结果还是米级，那说明系统还停留在 place/reference retrieval 层，不应该被包装成 final localization。

---

# 3. 从零开始的新项目定义

我建议把项目命名为：

> **Map-Conditioned Foundation Feature Selection for Visual Localization**

简称可以是：

```text
MC-FFS
```

或者更论文风格：

> **LoFSelect: Localizable Foundation Feature Selection for Prior-Map Visual Localization**

核心任务定义：

给定：

```text
1. query image Iq
2. prior map M
3. frozen foundation model Φ
4. candidate hypotheses H = {h_i}
```

学习一个 feature selector：

$$
S_\theta(\Phi(I)) = (z, m, \sigma)
$$

其中：

```text
z: compact selected feature
m: spatial utility / evidence mask
σ: uncertainty / risk
```

目标是：

> **选择出的 feature 能在先验地图条件下，对候选假设进行可靠验证，排除 hard negative，并把正确或可被下游 solver 修正的候选排到前面。**

注意，这里的目标不是：

```text
直接从 RADIO feature 做 PnP
直接做 continuous feature-metric refinement
直接超过 MASt3R / GS-CPR
```

而是：

```text
学到一种 map-conditioned localization evidence。
```

---

# 4. Candidate hypothesis 到底是什么

新方法里，candidate hypothesis 是：

> **一个可以被 query-map evidence 判断真假的定位假设。**

它不一定是 pose，也不一定是图像。

可以分成四类。

---

## 4.1 Place hypothesis

```text
h_i = map cell / room / corridor segment / reference cluster
```

任务：

```text
query 属于哪个地图区域？
```

这是 coarse localization / VPR 层。AnyLoc 证明 off-the-shelf self-supervised foundation features 可以作为 universal VPR substrate，SALAD 也用 DINOv2 和 dustbin cluster 丢弃 non-informative local features来提升 VPR。([arXiv](https://arxiv.org/abs/2308.00688?utm_source=chatgpt.com))

这类 hypothesis 可以是米级，但只能用于候选生成。

---

## 4.2 Reference-pose hypothesis

```text
h_i = (I_ref_i, T_ref_i)
```

任务：

```text
这个 reference image / reference pose 是否是 query 的好候选？
```

它比 place 更细，但仍不一定是最终 pose。它适合 Cambridge / 7-Scenes / HLoc top-K 这类 public protocol。

---

## 4.3 Rendered-pose hypothesis

```text
h_i = T_i ∈ SE(3)
```

给定 $T_i$，从先验地图渲染：

```text
RGB
depth
normal
world-position
selected feature map
visibility mask
```

任务：

```text
这个 pose 是否解释 query？
```

这才更接近精确 map-based localization。InLoc 的经典流程就是 retrieval 候选、dense matching 和 view synthesis pose verification；这说明“多候选 + verification”是视觉定位中的成熟范式，而不是随便造出来的 ranking 任务。([arXiv](https://arxiv.org/abs/1803.10368?utm_source=chatgpt.com))

---

## 4.4 Correspondence / solver hypothesis

```text
h_i = 一组 2D-3D matches
h_i = 一个 PnP pose
h_i = 一个 photometric refinement proposal
```

任务：

```text
这组 matches / 这个 solver proposal 可靠吗？
```

这里 selected feature 可以作为：

```text
match confidence
pose verification evidence
failure prediction
risk calibration
```

不一定负责生成 matches。

---

# 5. 新系统的整体结构

我建议系统分成四个模块。

```text
A. Candidate Generator
B. Foundation Feature Selector
C. Map-Conditioned Hypothesis Verifier
D. Fixed Downstream Solver / Refiner
```

---

## A. Candidate Generator：候选生成器

这个模块不要作为创新点。直接使用已有定位路线产生候选：

```text
HLoc / retrieval + matching
NetVLAD / SALAD / AnyLoc retrieval
ACE / scene coordinate regression
VO / SLAM prior
local pose lattice around an init pose
rendered pose candidates
```

HLoc 这类 pipeline 本来就是 retrieval、feature matching、SfM/PnP 的强视觉定位基线；GS-CPR 则代表了 3DGS render + MASt3R correspondence + refinement 的强 CPR 路线。([arXiv](https://arxiv.org/abs/2205.15761?utm_source=chatgpt.com))

我们的新方法不替代 candidate generator，而是回答：

> **在这些候选里，哪些是真的、哪些是 hard false positive？**

---

## B. Foundation Feature Selector：基础特征选择器

输入：

```text
raw VFM token bank F = Φ(I)
```

输出：

```text
z: D维 localizable descriptor，D=32/64
m: spatial utility map
σ: uncertainty / risk map
g: channel/layer gates
```

结构建议：

```text
raw token bank
  → layer/scale gate
  → channel group gate
  → low-rank / 1×1 projection
  → L2 normalization
  → spatial utility head
  → uncertainty head
```

形式化：

$$
z(u)=\mathrm{Norm}\left(W \sum_l \alpha_l(u) \left(g_l \odot F_l(u)\right)\right)
$$

其中：

```text
α_l: layer gate
g_l: channel group gate
W: compact projection
u: image/token position
```

关键是：**选择器必须可解释、可消融。**

不能只是一个大 MLP。否则它不是 feature selection，而是黑箱 descriptor learning。

---

## C. Map-Conditioned Hypothesis Verifier：地图条件假设验证器

对每个候选 $h_i$，构造证据：

```text
selected feature consistency
photometric consistency
depth / normal / geometry consistency
visibility consistency
uncertainty / risk
candidate prior
```

统一评分：

$$
s_i = G_\phi(E_{\text{feat}}, E_{\text{geom}}, E_{\text{photo}}, E_{\text{risk}})
$$

其中 selected feature evidence 可以是：

```text
patch-level local matching score
top-M correspondence score
correlation entropy
render-query selected feature residual
feature inlier ratio
utility-weighted similarity
```

不要用简单 global average pooling。当前很多负结果本质上就是 global pooled foundation feature 不能做 reference-pose ranking。

---

## D. Fixed Downstream Solver / Refiner：固定后端

后端可以是：

```text
PnP-RANSAC + LM
photometric alignment
RGB-D / depth alignment
GS-CPR-style render + matching
HLoc / ACE / SCR 后端
```

但它不是本文创新点。

selected feature 的作用是：

```text
1. 选择更好的候选；
2. 拒绝 hard false positive；
3. 提供 risk；
4. 将 top-K 送入 fixed solver。
```

Refinement 当然可以不用 selected feature。你说得对：如果有高质量 RGB/geometry map，直接优化 photometric error 或 depth error 都可以：

$$
E_{\text{photo}}(T)=\sum_u \rho(I_q(u)-R_{\text{rgb}}(M,T)(u))
$$

selected feature 不是要替代 photometric refinement，而是要补足 photometric / sparse matching 在难例中的弱点：

```text
重复结构
弱纹理
外观变化
曝光变化
render-real domain gap
retrieval false positives
```

---

# 6. 显式 3D 信息应该怎么用

我建议用显式 3D 信息，但不要一上来端到端重建 feature field。

正确顺序是：

```text
先学习定位可用子空间
再验证 mapability
再把 selected feature 嵌入 3D map
最后弱联合优化
```

---

## 6.1 Offline map construction

先验地图可以是：

```text
SfM point cloud
3DGS
mesh
RGB-D map
posed reference image set
```

每个 3D primitive 存：

```text
position X
visibility list
RGB / appearance
depth / normal
selected feature mean z_X
selected feature variance v_X
uncertainty
```

selected feature 聚合：

$$
z_X = \frac{\sum_i w_i z_i(u_i)}{\sum_i w_i}
$$

$$
v_X = \mathrm{Var}\{z_i(u_i)\}
$$

权重 $w_i$ 来自：

```text
visibility
viewing angle
depth consistency
spatial utility m(u)
uncertainty σ(u)
track consistency
```

这一步叫：

```text
selected-feature lifting
```

不是原始 RADIO feature reconstruction。

---

## 6.2 Online query-map scoring

测试时：

```text
query image
→ raw VFM feature
→ selector
→ query selected feature z_q
```

每个候选 pose：

```text
map render
→ selected feature map R_z(M,T_i)
→ depth / normal / visibility
→ verifier
→ score s_i
```

这样训练和测试统一为：

```text
query selected feature ↔ map selected feature
```

而不是训练时用 2D cache，测试时换另一套流程。

---

# 7. 训练流程：从零设计

我建议分五个阶段。

---

## Stage 0：Feature Localizability Audit

不训练，先评估 raw VFM tokens。

比较：

```text
raw RADIO final layer
raw RADIO middle layers
DINOv2
random projection
PCA
teacher reconstruction projection
```

任务：

```text
reference-pose ranking
rendered-pose ranking
3D track repeatability
hard negative separability
```

输出：

```text
layer utility
channel utility
spatial utility
mapability variance
```

目的：

> 先确认 foundation features 里是否真的有可用定位信号。

如果 raw RADIO 单独很差，但某些 layer/channel 有信号，那么 feature selection 有意义。  
如果所有 raw features 都没有可用信号，就不应该继续强做方法论文。

---

## Stage 1：2.5D Localizable Subspace Learning

训练 selector，但不训练 3D map。

监督来自几何：

```text
same 3D track positives
same-scene hard negatives
reference pose distance
rendered pose candidate distance
solver basin label
```

损失：

$$
L =
L_{\text{track}}
+
L_{\text{hard-neg}}
+
L_{\text{pose-rank}}
+
L_{\text{basin}}
+
L_{\text{sparse}}
+
L_{\text{calib}}
$$

其中：

### Track consistency

$$
L_{\text{track}}=\sum \|z_i(u_i)-z_j(u_j)\|
$$

同一个 3D 点跨视角 feature 接近。

### Hard negative contrast

同场景相似但错误的位置要分开：

$$
L_{\text{hard-neg}}
=
-\log
\frac{\exp(sim(z,z^+)/\tau)}
{\exp(sim(z,z^+)/\tau)+\sum_j\exp(sim(z,z_j^-)/\tau)}
$$

### Pose ranking

候选 pose 代价：

$$
c_i=d(T_i,T^\*)
$$

模型 score：

$$
s_i
$$

soft label：

$$
q_i=\mathrm{softmax}(-c_i/\tau_c)
$$

prediction：

$$
p_i=\mathrm{softmax}(s_i/\tau_s)
$$

loss：

$$
L_{\text{pose-rank}}=KL(q||p)
$$

### Basin classification

$$
y_i = 1[T_i \in \mathcal B_A(T^\*)]
$$

其中 $\mathcal B_A$ 是固定下游 solver 的成功 basin，最好通过实测得到，不只是几何阈值。

---

## Stage 2：Selected Feature Lifting / Mapability

将 Stage 1 选出的 feature 写入 3D map。

验证：

```text
track variance
primitive variance
render-query residual
rendered pose ranking retention
storage dimension
uncertainty calibration
```

目标是证明：

> selected feature 不是只在 2D cache 上有效，而是可以成为地图特征。

这一步非常关键。GSFF 已经证明 3DGS feature field + 2D encoder 共同 embedding 可以用于 visual localization；如果本项目不证明 selected feature mapability，就很容易被认为只是一个 2D reranker。([arXiv](https://arxiv.org/abs/2507.23569?utm_source=chatgpt.com))

---

## Stage 3：Map-Conditioned Hypothesis Verification

使用 selected 3D map feature 做候选验证。

输入：

```text
query selected feature
rendered selected map feature
rendered depth / normal / visibility
photometric residual
candidate prior
```

输出：

```text
candidate posterior
risk score
```

训练：

```text
pose-distance ranking
solver-basin classification
hard false positive suppression
calibration
```

---

## Stage 4：Hard-case Specialization

构建 hard-case benchmark：

```text
retrieval top1 wrong but topK right
repeated corridors / similar rooms
weak texture
photometric false positive
near-identity false positive
render-real domain gap
PnP high-inlier but wrong pose
```

训练和评估重点：

```text
hard false accept rate
top-K solver basin recall
risk-coverage
catastrophic error reduction
```

---

## Stage 5：Fixed Solver Handoff

选 top-K 候选后，交给固定后端：

```text
photometric/depth refinement
PnP-RANSAC
GS-CPR-style external CPR
HLoc / ACE solver
```

评价：

```text
final median translation / rotation
success@5cm/2°
success@10cm/5°
success@25cm/10°
catastrophic failure rate
runtime
```

如果 refinement 不提升，也没关系，但必须诚实说：

```text
本方法主要解决 hypothesis verification / hard false positive rejection。
```

---

# 8. 为什么这个方案比当前项目更清晰

新方案把每个模块的职责拆清楚：

```text
Foundation feature:
    提供通用视觉/结构/语义/潜在几何信号。

Feature selector:
    从 raw tokens 中选出对定位假设验证有用的子空间。

Explicit 3D map:
    提供 metric geometry、visibility、depth、pose-conditioned rendering。

Hypothesis verifier:
    判断候选地图/位姿/对应是否可信。

Fixed solver:
    完成最终 metric pose estimation 或 refinement。
```

这避免了之前的混乱：

```text
不是让 foundation feature 直接做 PnP；
不是让 feature reconstruction 自己解决定位；
不是把 photometric refinement 和 feature selection 混成一个目标；
不是用米级 reference pose ranking 冒充定位；
不是用历史 coarse/fine cache 冒充特征选择。
```

---

# 9. 评估体系：怎么证明特征选择有效

必须过五道门。

---

## Gate 1：Feature utility

证明 selected feature 比 raw foundation features 更适合定位候选判断。

比较：

```text
raw RADIO
raw DINOv2
PCA / random projection
teacher reconstruction
selected feature
selected feature without selection sparsity
```

指标：

```text
top1 candidate pose error
NDCG / Spearman / Kendall
basin recall@K
oracle gap
hard-negative AUC
```

---

## Gate 2：Causal selection

证明性能来自“选出来的特征”。

实验：

```text
remove high-utility channel groups
remove low-utility channel groups
mask high-utility spatial regions
mask low-utility spatial regions
shuffle query features
shuffle map features
wrong-scene features
metadata-only scorer
```

如果 high-utility 被移除后性能显著下降，而 low-utility 被移除影响小，才说明 selection 有因果意义。

---

## Gate 3：Mapability

证明 selected feature 能进入 3D map。

指标：

```text
multi-view track variance ↓
3D primitive variance ↓
render-query feature residual ↓
rendered-hypothesis ranking retention ↑
feature dimension / storage ↓
uncertainty calibration ↑
```

---

## Gate 4：Hard-case utility

证明它解决已有方法解决不了的难例。

hard set：

```text
retrieval false positive
similar room / corridor
weak texture
photometric ambiguity
near-identity trap
high PnP inlier but wrong pose
```

指标：

```text
hard false accept rate ↓
top-K basin recall ↑
risk-coverage AUC ↑
catastrophic error rate ↓
```

---

## Gate 5：Final localization

最终定位必须不是米级。

不同场景可设不同目标：

```text
Indoor prior-map:
    median < 10cm / 5°
    success@10cm/5° 提升
    hard-case catastrophic failure 降低

Outdoor Cambridge-like:
    median 至少达到 decimeter-level，且比 reference retrieval / HLoc init hard cases 更稳

Candidate verification:
    top-K 中包含 solver-success candidate 的概率显著提升
```

如果最终只能做到米级，那只能叫 place recognition 或 reference-pose retrieval，不能叫 precise localization。

---

# 10. 本项目的真正目标应该怎么写

我建议目标写成：

> **We aim to learn a compact, interpretable, and mapable subspace from frozen vision foundation features that provides robust map-conditioned evidence for localizing hard visual queries. The selected features are not intended to replace geometric or photometric solvers; instead, they verify and rank localization hypotheses, reject hard false positives, and provide calibrated handoff to fixed downstream localization pipelines.**

中文：

> 本项目目标是从 frozen vision foundation features 中学习一个紧凑、可解释、可嵌入地图的特征子空间，使其能够在先验地图条件下为困难 query 提供可靠定位证据。该特征不替代几何或光度 solver，而是用于验证和排序定位假设、拒绝 hard false positives，并为固定后端定位流程提供可靠候选和风险估计。

这个目标比“从 RADIO 中选定位特征”更严谨，也比“做 SOTA CPR”更可行。

---

# 11. 和已有工作的差异

新方案的位置应该这样解释：

```text
AnyLoc / SALAD:
    证明 foundation features 可以做 VPR / place recognition。
    但它们主要是 image-level retrieval，不解决 map-conditioned pose hypothesis verification。

InLoc / pose verification:
    证明多候选 pose verification 对室内定位很重要。
    但它们不是从 foundation model dense tokens 中学习可建图的 feature selection。

GS-CPR:
    用 3DGS RGB/depth + MASt3R 做强 correspondence-based refinement。
    但它不研究 foundation feature 子空间选择，也不把 selected feature 作为地图特征。

GSFF:
    学 3DGS feature field 和 2D encoder 的共同 embedding。
    但你的目标是从 frozen VFM token bank 中选择 localizable subspace，并解释哪些 feature 对定位有用。
```

这条定位比当前项目更清楚。([arXiv](https://arxiv.org/abs/2308.00688?utm_source=chatgpt.com))

---

# 12. 具体实施计划

## Phase 1：干净重建数据与输入

```text
1. 不使用 coarse_sem / fine_geo。
2. 提取 raw RADIO / C-RADIO spatial tokens。
3. 提取 raw DINOv2 spatial tokens作为对照。
4. 保留多层 / 多尺度 token bank。
5. 对所有 token 做统一 normalization、whitening、resolution alignment。
```

输出：

```text
raw_vfm_tokens/
    scene/
        image_id.npz:
            radio_layer_x
            dinov2_layer_y
            camera_pose
            intrinsics
```

---

## Phase 2：建立候选假设库

三类候选都要有：

```text
1. Place/reference candidates:
   retrieval top-K reference images / poses

2. Rendered pose candidates:
   around GT / init / retrieval pose 的 SE(3) hypotheses

3. Hard negatives:
   similar room, corridor, facade, near-identity trap, photometric false positive
```

每个候选记录：

```text
candidate_type
pose
submap_id
reference_image
pose_error
solver_success_label
depth/render availability
hard_case_type
```

---

## Phase 3：训练 Feature Selector

训练：

```text
layer gate
channel group gate
projection
spatial utility
uncertainty
```

损失：

```text
track consistency
hard negative contrast
pose hypothesis ranking
basin classification
sparsity
calibration
```

不训练 3D map。

---

## Phase 4：构建 Selected Feature Map

用 selected feature 聚合到：

```text
SfM points
3DGS Gaussians
mesh/surfels
```

存储：

```text
mean z
variance
visibility count
uncertainty
```

渲染：

```text
selected feature map
uncertainty map
depth / normal / visibility
```

---

## Phase 5：训练 Map-Conditioned Verifier

输入：

```text
query selected feature
rendered selected map feature
rendered RGB/depth/normal
candidate prior
```

输出：

```text
hypothesis score
risk
accept/reject
```

目标：

```text
rank correct candidates
reject hard false positives
calibrate failure risk
```

---

## Phase 6：固定后端评估

后端不要训练：

```text
HLoc / PnP
photometric refinement
GS-CPR-style external baseline
depth alignment
```

只评估：

```text
selected features 是否改善候选选择、risk 和 final success。
```

---

# 13. 最后给一句最直接的结论

你说得完全对：**继续使用 `coarse_sem / fine_geo`，并且得到米级结果，不能支撑本项目目标。**

新的项目应该从头定义为：

```text
raw VFM token bank
→ localization-conditioned feature selection
→ selected feature lifting into explicit 3D map
→ map-conditioned hypothesis verification
→ fixed solver handoff
```

核心 claim 不是：

```text
RADIO 特征直接精细定位。
```

而是：

```text
从基础模型 dense tokens 中选择一个可建图、可解释、对先验地图定位假设验证有用的特征子空间，用来解决现有定位方法在 hard false positives、重复结构、弱纹理和 render-real gap 下的失败。
```

这个版本的逻辑更完整，和同领域术语也更一致：  
**feature selection、map-conditioned hypothesis verification、pose verification、selected feature map、hard negative rejection、solver handoff、risk calibration**。  
如果按这条线重建，项目才有可能从“工程探索”变成一篇有清晰问题和方法闭环的顶刊工作。

---

**Sources:**

- [AM-RADIO: Agglomerative Vision Foundation Model -- Reduce All Domains Into One](https://arxiv.org/abs/2312.06709?utm_source=chatgpt.com)

## Prompt:
2026/5/24 17:29:00

我重新新建了一个git 分支vfm，从之前的radio分支继承而来，需要彻底清理一下以前的旧主线，包括代码实现和实验配置。然后重新梳理新主线，干净整洁，之后实验也要完全重做。
最后明确本项目的评估协议和预期指标

## Response:
2026/5/24 17:29:01

我会把 `vfm` 分支当作一次彻底重启来设计：先隔离/下线旧 `radio` 的 Stage、POFD、CPR 历史路径，再重建干净的新主线、代码边界、实验协议和验收指标。

## Response:
2026/5/24 17:29:15

> **检查 GitHub URL 连接**
>
>
> 已思考 4m 1s

## Response:
2026/5/24 17:33:03

你的 `vfm` 分支应该做一次**真正意义上的 reset**，不要再从 `radio` 分支的旧叙事上“修补”。我建议把新分支定义成一个干净项目：

> **VFM-MapLoc：Foundation Feature Selection for Prior-Map Visual Localization**  
> 从 frozen vision foundation model 的原始 dense tokens 中，选择一个紧凑、可解释、可嵌入先验地图的 localizable feature subspace，用于地图条件下的定位假设验证、hard negative 排除和下游 solver handoff。

旧 `radio` 分支 README 仍然把主线写成 `retrieval/init -> scene feature render -> dense correspondence / feature flow -> geometric solve -> iterative refinement`，并且当前 strongest benchmark 仍含 oracle retrieval，这说明它的工程和叙事都不再适合新目标。([GitHub](https://github.com/Arthurshen926/ICLPose/tree/radio)) 新 `vfm` 分支应该停止继承这套主线。

---

# 1. 新分支的唯一主 claim

不要再写：

> RADIO-guided feature-field reconstruction and visual localization.

也不要写：

> feature refinement / pose refinement / SOTA CPR.

新的主 claim 应该是：

> **We learn a compact, interpretable, and mapable subspace from frozen vision foundation model dense tokens for prior-map visual localization. The selected features provide map-conditioned evidence for hypothesis verification, hard-negative rejection, and reliable handoff to fixed downstream solvers.**

中文：

> 我们从 frozen 视觉基础模型 dense tokens 中学习一个紧凑、可解释、可建图的定位可用子空间。该特征用于先验地图条件下的定位假设验证、hard negative 排除，以及向固定后端 solver 的可靠候选交接。

这个 claim 的重点是：

```text
raw VFM tokens
→ feature selection
→ map-conditioned hypothesis verification
→ selected-feature mapability
→ fixed-solver handoff
```

而不是：

```text
coarse_sem / fine_geo
→ candidate rerank
→ LoFTR / PnP fallback
→ pose refinement
```

---

# 2. 先清理旧主线：哪些必须移出主路径

## 2.1 旧概念全部下线

这些概念不要出现在新 README、新 config、新实验表主列里：

```text
coarse_sem
fine_geo
DCFF v5/v6/v7 主线
Stage2 / Stage3 / Stage4
POFD Stage4 continuous CPR
query-student as main method
pose-adapted as main method
concat_loc_*
teacher-query baseline
oracle retrieval benchmark
render-LoFTR refinement as method
learned WLS/GN pose update as main claim
```

它们可以留在：

```text
legacy/
docs/archive/
```

但不能再作为当前方法接口。

原因是：`coarse_sem / fine_geo` 这类名字来自旧 RADIO 中间层缓存，不是干净的 raw foundation feature bank。它们会让方法看起来像“历史工程特征调参”，而不是 foundation feature selection。

---

## 2.2 旧代码不要直接删光，先隔离

建议做三层处理。

### A. 保留为公共工具的代码

这些可以保留，但必须改成中性命名：

```text
camera / pose / SE3 utilities
COLMAP / SfM / dataset loaders
projection / rendering utilities
evaluation metrics
3D map IO
HLoc / retrieval cache parsers
```

放到：

```text
vfm_loc/common/
vfm_loc/datasets/
vfm_loc/geometry/
vfm_loc/rendering/
vfm_loc/evaluation/
```

### B. 移入 legacy 的代码

这些移动到：

```text
legacy/radio_pipeline/
```

包括：

```text
pose_refine/
feature_field/dcff/
feature_retrieval/old_*
feature_extract/tools/train_nvs_pose_feature_adapter.py
feature_extract/localizability/旧实验入口
configs/pofd_*
configs/joint_radio_*
configs/concat_loc_*
docs/superpowers/plans/2026-05-*
```

### C. 删除或禁用的内容

这些不要留在主 configs 里：

```text
旧 smoke configs
绝对路径 configs
oracle retrieval 默认配置
缓存字段依赖 external PnP / LoFTR score 的训练 config
teacher reconstruction 作为主目标的 config
```

---

# 3. 新项目目录结构

建议新分支顶层只保留一个主包：

```text
vfm_loc/
  README.md
  pyproject.toml 或 setup.cfg

  vfm_loc/
    foundations/
      extract_tokens.py
      backbones.py
      normalize.py

    datasets/
      colmap_scene.py
      cambridge.py
      seven_scenes.py
      oldhospital.py
      split.py

    maps/
      sfm_map.py
      gaussian_map.py
      selected_feature_map.py
      map_lifting.py
      map_render.py

    hypotheses/
      reference_bank.py
      pose_lattice.py
      hard_negative_mining.py
      solver_basin.py
      schema.py

    selection/
      selector.py
      gates.py
      projection.py
      uncertainty.py

    verification/
      evidence.py
      patch_matcher.py
      hypothesis_scorer.py
      fusion.py

    losses/
      rank.py
      contrastive.py
      mapability.py
      calibration.py
      sparsity.py

    evaluation/
      metrics.py
      protocols.py
      reports.py
      visualization.py

    solvers/
      pnp.py
      photometric.py
      handoff.py

  configs/
    00_extract/
    01_audit/
    02_selection/
    03_mapability/
    04_hypothesis_verification/
    05_solver_handoff/

  experiments/
    oldhospital/
    cambridge/
    seven_scenes/

  docs/
    METHOD.md
    PROTOCOL.md
    RESULTS_TEMPLATE.md
    ABLATIONS.md
    ARCHIVE_NOTES.md

  tests/
    test_tokens.py
    test_hypothesis_schema.py
    test_no_oracle_leakage.py
    test_metrics.py
    test_map_lifting.py
```

主入口只保留这些：

```bash
python -m vfm_loc.foundations.extract_tokens
python -m vfm_loc.hypotheses.build_bank
python -m vfm_loc.selection.train_selector
python -m vfm_loc.maps.lift_selected_features
python -m vfm_loc.verification.train_verifier
python -m vfm_loc.evaluation.eval_protocol
python -m vfm_loc.evaluation.report
```

---

# 4. 新方法的完整数据流

## 4.1 Offline map construction

```text
posed reference images
  ↓
frozen VFM token extraction
  ↓
feature selection / projection
  ↓
multi-view selected-feature lifting
  ↓
selected-feature prior map
```

地图里每个 3D primitive 存：

```text
position X
visibility list
selected feature mean z_X
selected feature variance v_X
utility statistics
uncertainty statistics
view-count
optional RGB / depth / normal
```

---

## 4.2 Online query localization

```text
query image
  ↓
frozen VFM token extraction
  ↓
same selector Sθ
  ↓
query selected feature z_q, utility m_q, uncertainty σ_q

prior map + candidate hypotheses
  ↓
render / retrieve / project candidate evidence
  ↓
hypothesis verifier
  ↓
ranked candidates + risk

top-K
  ↓
fixed downstream solver, optional
```

关键点：

> **训练和测试必须使用同一套 selected feature 流程。**

不能训练时用 cached pose-adapted feature，测试时换 reference pooled descriptor；不能训练时用 rendered candidate，测试时只用 global retrieval；不能训练时用 external LoFTR/PnP score 作为 label，又 claim feature selection。

---

# 5. Candidate hypothesis 的统一定义

新项目中，candidate hypothesis 统一定义为：

> **一个可被 query-map evidence 判断真假的定位假设。**

它可以有四类。

## 5.1 Place hypothesis

```text
h = map cell / room / corridor segment / reference cluster
```

作用：

```text
coarse candidate generation
```

指标可以是米级，但只作为 coarse stage。

## 5.2 Reference-pose hypothesis

```text
h = (reference_image, T_ref)
```

作用：

```text
public benchmark / retrieval reranking
```

用于 Cambridge / 7-Scenes / HLoc top-K。

## 5.3 Rendered-pose hypothesis

```text
h = T_i ∈ SE(3)
```

从先验地图渲染：

```text
selected feature
RGB
depth
normal
world-position
visibility
```

作用：

```text
map-conditioned pose verification
```

这是精确定位最核心的 hypothesis 类型。

## 5.4 Solver hypothesis

```text
h = PnP proposal / correspondence set / photometric refinement proposal
```

作用：

```text
accept/reject
risk calibration
handoff validation
```

---

# 6. 新方法模块

## 6.1 Foundation Token Bank

从 frozen model 提取原始 tokens：

```text
RADIO / C-RADIO spatial tokens
DINOv2 tokens
optional SigLIP / CLIP dense tokens
```

不要再叫：

```text
coarse_sem
fine_geo
```

统一叫：

```text
vfm_tokens[layer, scale]
```

每个 token cache schema：

```python
{
  "image_id": str,
  "scene_id": str,
  "tokens": {
      "radio_lX": FloatTensor[H, W, C],
      "dinov2_lY": FloatTensor[H, W, C],
  },
  "pose": FloatTensor[4, 4],
  "intrinsics": FloatTensor[3, 3],
  "image_size": [H_img, W_img],
}
```

---

## 6.2 Localizable Feature Selector

输入：

```text
raw VFM token bank
```

输出：

```text
z: compact feature, D=32 或 64
m: spatial utility map
σ: uncertainty / risk
g: channel / layer gates
```

推荐结构：

```text
layer gate
channel group gate
low-rank projection
L2 normalize
spatial utility head
uncertainty head
```

公式：

$$
z(u)=\mathrm{Norm}\left(W \sum_l \alpha_l(u)\,(g_l \odot F_l(u))\right)
$$

必须有 sparsity，否则不是 feature selection。

损失里加入：

```text
group lasso / L0-style sparsity
utility entropy
channel diversity
```

---

## 6.3 Map-Conditioned Hypothesis Verifier

对候选 $h_i$，提取证据：

```text
E_feat: selected feature consistency
E_geom: depth / normal / visibility consistency
E_photo: optional photometric consistency
E_prior: retrieval / candidate prior
E_risk: uncertainty / ambiguity
```

输出：

```text
score s_i
risk r_i
```

建议 verifier 不要一开始做大 Transformer。先用：

```text
patch-level local evidence aggregation
top-M local matches
utility-weighted similarity
correlation entropy
valid-pixel ratio
uncertainty-weighted residual
small MLP fusion
```

避免退化为黑箱。

---

## 6.4 Selected Feature Map

不要重建 full RADIO feature。只重建 selected feature：

$$
z_X = \frac{\sum_i w_i z_i(u_i)}{\sum_i w_i}
$$

$$
v_X = Var\{z_i(u_i)\}
$$

其中：

```text
w_i = visibility × view-angle weight × utility × uncertainty inverse × depth consistency
```

地图渲染输出：

```text
rendered selected feature R_z(M,T)
rendered variance R_v(M,T)
valid mask
optional RGB/depth/normal
```

---

# 7. 新训练流程

## Stage 0：Raw VFM audit

目的：

```text
证明原始 foundation features 是否有定位信号。
```

实验：

```text
raw RADIO
raw DINOv2
PCA projection
random projection
global pooled descriptor
patch-level descriptor
```

任务：

```text
reference-pose ranking
rendered-pose ranking
track repeatability
hard negative separability
```

通过标准：

```text
至少某些 layer / patch evidence 优于 random / PCA；
否则 feature selection 没有基础。
```

---

## Stage 1：2.5D localizable subspace learning

不训练地图，只训练 selector。

监督：

```text
3D track positives
same-scene hard negatives
pose-distance ranking
basin label
feature shuffle controls
metadata-only controls
```

损失：

$$
L =
L_{\text{track}}
+
L_{\text{hard-neg}}
+
L_{\text{rank}}
+
L_{\text{basin}}
+
L_{\text{sparse}}
+
L_{\text{calib}}
$$

目标：

```text
选出的 z 在候选验证上优于 raw tokens / reconstruction / metadata-only。
```

---

## Stage 2：Selected-feature mapability

冻结 selector，将 selected features 聚合到 3D map。

评估：

```text
track variance
primitive variance
render-query residual
rendered-pose ranking retention
storage dimension
uncertainty calibration
```

通过标准：

```text
selected feature variance < raw feature variance；
selected feature D <= 64；
rendered selected feature ranking 不明显退化。
```

---

## Stage 3：Map-conditioned hypothesis verification

用 selected-feature map 训练 verifier。

候选包括：

```text
reference-pose candidates
rendered-pose candidates
hard negatives
near-identity traps
photometric false positives
```

目标：

```text
hard false accept rate 下降；
top-K basin recall 提升；
ranking NDCG / Spearman 提升。
```

---

## Stage 4：Fixed solver handoff

只用固定后端：

```text
PnP-RANSAC + LM
photometric / depth alignment
HLoc / GS-CPR-style external solver
```

评估 selected feature 是否改善：

```text
candidate selection
accept/reject
risk prediction
hard-case final success
```

不要训练 solver，也不要把 solver 当方法创新点。

---

# 8. 必须废弃的旧实验配置

在 `configs/` 下只保留新命名，旧的全部移走。

## 删除 / 移到 archive

```text
configs/dcff_*
configs/joint_radio_*
configs/pofd_stage*
configs/concat_loc_*
feature_extract/configs/pofd_*
pose_refine/configs/*
feature_retrieval/* old trainable selector configs
```

## 新 configs

```text
configs/00_extract/
  radio_tokens.yaml
  dinov2_tokens.yaml

configs/01_audit/
  raw_tokens_reference_ranking.yaml
  raw_tokens_rendered_pose_ranking.yaml
  track_repeatability.yaml

configs/02_selection/
  selector_track_hardneg.yaml
  selector_pose_rank.yaml
  selector_basin_rank.yaml

configs/03_mapability/
  lift_selected_features.yaml
  eval_selected_mapability.yaml

configs/04_verification/
  rendered_pose_verifier.yaml
  reference_pose_verifier.yaml
  hard_case_verifier.yaml

configs/05_handoff/
  pnp_handoff.yaml
  photometric_handoff.yaml
  risk_coverage.yaml
```

---

# 9. 新 README 应该怎么写

README 第一屏只写：

```text
# VFM-MapLoc

Foundation feature selection for prior-map visual localization.

We study how to select compact, interpretable, and mapable feature subspaces
from frozen vision foundation model dense tokens for map-conditioned
localization hypothesis verification.

The method is not a camera pose refinement network.
The method is not an oracle candidate reranker.
The method does not use external matcher/PnP scores as feature-selection labels.
```

主 pipeline：

```text
VFM token extraction
→ localizable feature selection
→ selected-feature lifting into prior map
→ map-conditioned hypothesis verification
→ fixed solver handoff
```

---

# 10. 评估协议

评估协议必须分层，否则米级 retrieval 和厘米级定位会混在一起。

---

## Protocol A：Feature Localizability Audit

目的：

```text
判断 raw foundation features 是否含定位信号。
```

输入：

```text
query image
candidate references / poses
raw tokens
```

指标：

```text
top1 candidate error
top5 oracle error
NDCG
Spearman / Kendall
hard-negative AUC
track repeatability
```

通过指标：

```text
raw VFM patch-level evidence > random / PCA；
如果 raw global pooled descriptor 是米级，不作为失败，只说明 global pooling 不够。
```

---

## Protocol B：Selection Causality

目的：

```text
证明选择器真的选择了有用特征。
```

必须做：

```text
high-utility channel removal
low-utility channel removal
high-utility spatial mask removal
low-utility spatial mask removal
query feature shuffle
map feature shuffle
wrong-scene feature
metadata-only scorer
```

通过指标：

```text
移除 high-utility evidence 导致明显 ranking drop；
移除 low-utility evidence 影响小；
shuffle 后性能显著下降；
metadata-only baseline 打不过 feature scorer。
```

---

## Protocol C：Rendered Pose Hypothesis Verification

目的：

```text
精确先验地图候选验证。
```

候选：

```text
q10 / q25 / q50 local pose hypotheses
retrieval init neighborhood
near-identity traps
photometric false positives
```

指标：

```text
top1 pose error
topK oracle gap
basin recall@1/5/10
hard false accept rate
NDCG
Spearman
risk-coverage
```

预期指标：

```text
q10:
  median top1 <= 5cm
  basin@1(10cm/5°) >= 0.90

q25:
  median top1 <= 10cm
  basin@1(25cm/10°) >= 0.85

q50:
  median top1 <= 20cm
  basin@1(25cm/10°) >= 0.75
  basin@5(25cm/10°) >= 0.90
  hard false accept rate 降低 >= 30% 相比 raw VFM / retrieval prior
```

注意：q50 top1 如果仍是米级，说明不是精确定位候选验证。

---

## Protocol D：Reference-Pose / Public Multi-Scene Ranking

目的：

```text
证明不是 OldHospital-only。
```

数据：

```text
Cambridge / 7-Scenes / OldHospital / ShopFacade
```

候选：

```text
retrieval top-K reference poses
HLoc top-K reference poses
```

指标：

```text
reference-pose top1 error
top5 oracle
NDCG
basin recall@K
hard-case recall
```

合理目标：

```text
至少 3 个场景上：
  selected feature scorer 优于 raw VFM global/patch baseline；
  selected feature scorer 优于 retrieval order 或至少提升 hard-case subset；
  top5 basin recall 提升 >= 10% absolute。
```

如果 reference-pose 本身米级，可以接受，但必须明确它是 coarse stage。

---

## Protocol E：Selected Feature Mapability

目的：

```text
证明 selected feature 可嵌入 3D 先验地图。
```

指标：

```text
track variance ↓
primitive variance ↓
render-query residual ↓
rendered-pose ranking retention ↑
storage dimension ↓
uncertainty calibration ↑
```

预期目标：

```text
D <= 64
track variance 比 raw VFM 降低 >= 20%
rendered-pose ranking retention >= 90% of 2.5D selected feature performance
selected map feature 优于 raw map feature
```

---

## Protocol F：Solver Handoff / Final Localization

目的：

```text
证明 selected feature 对现有定位后端有实际帮助。
```

固定 solver：

```text
PnP-RANSAC + LM
photometric / depth refinement
external GS-CPR-style solver
```

指标：

```text
final median translation / rotation
success@5cm/2°
success@10cm/5°
success@25cm/10°
catastrophic failure rate
risk-coverage AUC
runtime
```

目标：

```text
Indoor hard cases:
  success@10cm/5° 提升 >= 10% absolute
  catastrophic failure rate 降低 >= 30%
  median final error 不劣于 fixed solver baseline

If final solver does not improve:
  method claim 降级为 hypothesis verification / risk prediction
```

---

# 11. 本项目最终预期指标

## 最低可投稿指标

```text
1. 使用 raw RADIO / DINOv2 tokens，不使用 coarse_sem / fine_geo 旧缓存名。
2. selected feature D <= 64。
3. rendered-pose q10/q25 明显优于 raw VFM / PCA / random / reconstruction。
4. q50 top1 median <= 20cm 或 basin@5(25cm/10°) >= 0.90。
5. 至少 2 个场景有正结果。
6. selected feature mapability 成立：
   variance 降低 >= 20%，rendered ranking retention >= 90%。
7. hard false accept rate 相比 raw VFM 降低 >= 30%。
8. feature shuffle / metadata-only controls 不能解释性能。
```

---

## TRO / 顶刊目标

```text
1. 至少 3 个场景，其中至少 1 个 public benchmark。
2. 至少 2 种 hypothesis type：
   reference-pose + rendered-pose。
3. 至少 2 种 candidate generator：
   retrieval/HLoc + local pose lattice。
4. selected feature 在 hard cases 上稳定优于 raw VFM、retrieval prior、photometric score。
5. selected feature 可嵌入 3D map，并在 query-map same pipeline 中有效。
6. fixed solver handoff 至少带来 hard-case success 或 catastrophic failure reduction。
7. 提供 causal selection evidence：
   channel/spatial counterfactual ablations 成立。
```

---

## Stop 条件

如果出现以下情况，应停止“方法论文”路线：

```text
1. raw VFM tokens 完全没有任何 localizability signal；
2. selected feature 在 q10/q25 都打不过 raw VFM / PCA；
3. selected feature mapability 不成立；
4. feature shuffle 后性能仍然不掉；
5. metadata-only scorer 接近或超过 selected feature scorer；
6. 第二场景完全没有正结果；
7. q50 / hard cases 仍然米级。
```

如果触发 stop，可以改成：

```text
benchmark / diagnostic paper:
Why foundation features fail for prior-map localization.
```

---

# 12. 给 Codex 的清理目标

可以直接把下面这段作为 goal。

```text
Goal:
Reset the new vfm branch into a clean project:
VFM-MapLoc: Foundation Feature Selection for Prior-Map Visual Localization.

Main claim:
Learn a compact, interpretable, and mapable feature subspace from frozen raw VFM dense tokens for map-conditioned localization hypothesis verification.

Hard constraints:
- Do not use coarse_sem / fine_geo as method concepts.
- Do not use old Stage2/Stage3/Stage4 naming.
- Do not promote continuous CPR as the main method.
- Do not use external LoFTR/MASt3R/PnP scores as feature-selection labels.
- Do not use oracle retrieval in deploy protocol.
- Keep candidate generation and solver modules fixed baselines.
- Main method is feature selection + map-conditioned hypothesis verification.

Cleanup:
1. Move old radio pipeline code/configs/docs into legacy/radio_pipeline.
2. Create vfm_loc package with:
   foundations, datasets, maps, hypotheses, selection, verification,
   losses, evaluation, solvers.
3. Replace old configs with:
   00_extract, 01_audit, 02_selection, 03_mapability,
   04_verification, 05_handoff.
4. Write new README, METHOD.md, PROTOCOL.md.

Implementation:
1. Raw VFM token extraction:
   RADIO/C-RADIO and DINOv2 tokens.
2. Candidate hypothesis bank:
   place/reference/pose/solver hypotheses.
3. LocalizableFeatureSelector:
   layer gate, channel group gate, compact projection, spatial utility,
   uncertainty.
4. SelectedFeatureMap:
   multi-view lifting, mean/variance, rendering.
5. MapConditionedVerifier:
   feature, geometry, photometric, risk evidence.
6. Evaluation protocols:
   audit, selection causality, rendered-pose verification,
   public reference-pose ranking, mapability, solver handoff.

Primary metrics:
- q10 top1 <= 5cm, basin@1(10cm/5deg) >= 0.90
- q25 top1 <= 10cm, basin@1(25cm/10deg) >= 0.85
- q50 top1 <= 20cm or basin@5(25cm/10deg) >= 0.90
- hard false accept rate reduction >= 30%
- selected feature D <= 64
- mapability variance reduction >= 20%
- rendered ranking retention >= 90%
- at least 2 scenes positive, including one public or public-style scene

Deliverables:
- clean repo tree
- new configs
- new docs
- no old mainline references in README
- unit tests for hypothesis schema, token cache, no-oracle leakage,
  metric correctness, map lifting.
```

---

# 13. 最终判断

你现在新建 `vfm` 分支是正确的。这个分支不应该继续修补 `radio` 分支，而应该彻底变成：

```text
raw foundation tokens
→ localizable feature selection
→ selected feature map
→ map-conditioned hypothesis verification
→ fixed solver handoff
```

评估也必须从一开始分清：

```text
place-level 米级召回
hypothesis-level 分米级验证
final localization 厘米/分米级成功率
```

如果最终结果仍然是米级，那它只能叫 retrieval / place recognition，不能叫 prior-map visual localization。  
如果 selected feature 能在 q10/q25/q50、hard negatives、mapability、second scene 上成立，才有资格支撑“foundation feature selection for visual localization”的强 claim。

---

**Sources:**

- [GitHub - Arthurshen926/ICLPose at radio · GitHub](https://github.com/Arthurshen926/ICLPose/tree/radio)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)