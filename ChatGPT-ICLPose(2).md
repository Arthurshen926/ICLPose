从“**最容易中稿顶刊**”的角度，我建议把本项目彻底重构成一篇 **feature-centric localization paper**，而不是 camera pose refinement system paper。

最稳的方向不是：

> RADIO / POFD 直接做连续位姿 refinement，并打赢 GS-CPR / MASt3R / HLoc。

而是：

> **Which Foundation Features Localize?**
> 系统研究 dense foundation model features 的“定位可用性”，并提出一种面向 **pose / map hypothesis ranking** 的特征选择框架。

当前仓库的新进展正好支持这个转向：`feature_extract/localizability/` 已经实现了 selector、scorer、losses、candidate bank、metrics、mapability 等新边界；新文档也明确把主线定为 **POFD-FS: Pose-Observable Foundation Feature Selection**，核心任务是 `score(query, map, candidate_pose)`，而不是连续 sparse correspondence 或 feature-metric pose update。([GitHub][1])

---

# 1. 最容易中稿的论文定位

我建议论文定位为：

> **A benchmark-and-method paper on localization-usable subspaces of foundation model features.**

也就是：

```text
问题定义 + 评估协议 + 方法 + 系统性正负结论
```

而不是：

```text
又一个 pose refinement pipeline
```

原因很直接：

1. 当前 POFD continuous CPR / Stage4 已经被大量实验证明不够稳。HLoc real-init 下 POFD Stage4 是 no-op，RGB LoFTR render-at-init 也不稳定；文档里明确记录 OldHospital / ShopFacade 上 solver_success 和 accepted 都是 0。([GitHub][2])
2. 当前真正有正证据的是 **hypothesis ranking**。OldHospital controlled ranking 里，`pose-adapted + pair matcher r16` 明显优于 raw RADIO 和 query student：q10 `0.045m`、q25 `0.107m`、q50 `0.223m`；q50 top1 `0.719`、Spearman `0.586`，说明 adapted foundation feature 已经能形成一定 pose score surface。([GitHub][1])
3. 已有工作已经证明 foundation features 在 VPR / place recognition 上很强，例如 AnyLoc 使用 off-the-shelf self-supervised features 做 universal VPR，SALAD 使用 DINOv2 并通过 dustbin cluster 丢弃 non-informative local features；所以“foundation feature selection for localization”有明确背景，但现有工作主要停留在 image-level VPR，不是 3D pose-hypothesis localizability。([anyloc.github.io][3])
4. GS-CPR / GSFF 这类工作已经占据了 3DGS pose refinement 和 3D feature field localization 的强系统路线。GS-CPR 使用 3DGS 渲染 RGB/depth，并用 MASt3R 建立 2D-3D correspondence；GSFF 则学习 3DGS feature field 和 2D encoder 的共同 embedding。你的论文如果继续主打 CPR，会被迫和这些强系统正面对比，风险很高。([arXiv][4])

因此，最容易中稿的论文不是说：

> 我们提出 SOTA visual localization / CPR 系统。

而是说：

> 我们首次系统定义、评估并学习 foundation dense features 的 localization-usable subspace，证明其最适合的角色是 **candidate / pose-hypothesis discrimination**，而不是直接 sparse correspondence 或 continuous pose refinement。

这条路线更有理论深度，也更容易把当前仓库里的正负结果转化为论文贡献。

---

# 2. 论文主问题应该这样定义

不要笼统说“foundation feature 对定位有用”。这句话太宽，因为定位方法很多。

我建议定义四级 localizability：

| 层级      | 名称                            | 含义                               | 本文角色             |
| ------- | ----------------------------- | -------------------------------- | ---------------- |
| Level 1 | Place-localizability          | 能否识别正确地点 / 地图区域                  | 背景与辅助            |
| Level 2 | Hypothesis-localizability     | 能否在候选地图 / 候选位姿中选出正确或可修正候选        | **主任务**          |
| Level 3 | Correspondence-localizability | 能否产生 2D-2D / 2D-3D matches，接 PnP | 下游验证             |
| Level 4 | Differential-localizability   | 能否支持连续 feature-metric refinement | 负结果 / limitation |

你的论文主打 **Level 2**。

形式化定义：

给定 query 图像 (I_q)、地图 (M)、候选集合：

[
C = {h_i}_{i=1}^K
]

其中 (h_i) 可以是：

```text
map cell
reference image / reference pose
local pose perturbation
rendered candidate pose
```

学习一个分数函数：

[
s_\theta(I_q, M, h_i)
]

使得 score 排序接近真实几何 utility：

[
d(h_i, h^*) < d(h_j, h^*)
\Rightarrow
s_\theta(h_i) > s_\theta(h_j)
]

更强一点，可以定义为 **basin-aware utility**：

[
h_i \in \mathcal{B}*A(h^*)
\Rightarrow
s*\theta(h_i) > s_\theta(h_j \notin \mathcal{B}_A)
]

其中 (\mathcal{B}_A) 是下游 solver 或 localization pipeline 的成功 basin。

这句话非常关键：

> **foundation feature 不需要自己完成厘米级 pose refinement；它需要把“值得交给几何 solver 或下游 pipeline 的候选”排到前面。**

这就是最适合 RADIO / DINO / SigLIP / C-RADIO 这类 foundation dense features 的定位角色。

---

# 3. 最推荐的论文题目与 claim

## 题目方向

```text
Which Foundation Features Localize?
A Benchmark and Selection Framework for Hypothesis-Based Visual Localization
```

或者：

```text
POFD-FS: Pose-Observable Foundation Feature Selection for Visual Localization
```

## 主 claim

建议写成：

> **Dense foundation features contain localization-usable subspaces, but their usefulness depends on the localization paradigm. We show that they are better suited for basin-aware pose-hypothesis ranking than direct sparse correspondence or continuous pose refinement, and we propose POFD-FS to select compact, mapable, and interpretable feature subspaces for this role.**

中文：

> Dense foundation features 中确实存在定位可用子空间，但这种可用性依赖定位范式。我们证明它们更适合做 basin-aware pose-hypothesis ranking，而不是直接 sparse correspondence 或连续 pose refinement；并提出 POFD-FS，从 frozen foundation features 中选择 compact、可解释、可嵌入 3D map 的定位子空间。

---

# 4. 论文贡献应该设计成 4 个

## Contribution 1：Localization utility taxonomy

提出一个 taxonomy：

```text
place utility
hypothesis utility
correspondence utility
differential utility
```

然后明确结论：

> Foundation dense features 的强项不一定是 differential pose refinement，而是 hypothesis discrimination。

这能把之前 Stage4 no-go 变成有价值结论，而不是失败记录。

## Contribution 2：Foundation Feature Localizability Benchmark

建立一个 benchmark，不只是一个方法。

候选集包括：

```text
1. controlled local-lattice pose banks: q10 / q25 / q50
2. reference-pose candidate banks: public datasets, no render required
3. same-scene hard negative banks
4. 3D feature-field rendered candidate banks
5. real-init stress banks: HLoc / NetVLAD / external initializers
```

这个 benchmark 的意义是：

> 评估 foundation features 是否能把正确或可修正候选排到前面。

当前仓库已经有 q10/q25/q50 localizability banks 和 standardized report 工具，这正好是雏形。([GitHub][1])

## Contribution 3：POFD-FS 方法

提出：

```text
frozen foundation feature
→ channel / layer / spatial selector
→ compact localization descriptor
→ pose-hypothesis scorer
→ basin-aware ranking loss
```

当前 `LocalizationFeatureSelector` 已经实现了 grouped channel gate、1x1 compact projection、spatial utility head、uncertainty head 和 L2-normalized selected feature；`PoseHypothesisScorer` 支持 same-pixel、local-correlation 和 pair_matcher_local scoring。([GitHub][5])

## Contribution 4：Systematic findings

这部分非常适合顶刊：

```text
1. raw foundation features 有一定 localization signal，但不足以直接 fine localization；
2. teacher reconstruction 不等价于 localization feature；
3. confidence / Fisher / feature health 不保证 pose ranking；
4. low-res large-radius hypothesis scoring 比 high-res small-radius 更适合中等 basin；
5. selected features 在 q10/q25 上显著优于 raw/reconstruction，在 q50 上接近 promote threshold；
6. continuous CPR no-go 说明 foundation features 当前更适合 hypothesis utility，而不是 differential utility。
```

这类“正结果 + 负结果 + 原因分析”非常适合 journal，不一定要求单表 SOTA。

---

# 5. 最容易中稿的最终方法架构

我建议最终方案是一个三层架构。

---

## Layer A：Feature Localizability Selector

输入：

```text
RADIO / C-RADIO / DINOv2 / SigLIP dense features
```

输出：

```text
z_q: compact localization feature
m_q: spatial utility
σ_q: uncertainty / risk
g: channel or layer gates
```

模块：

```text
channel group gate
optional layer gate
1x1 / low-rank projection to 32D or 64D
spatial utility head
uncertainty head
```

第一版可以继续用当前仓库里已有的 RADIO / query-student / pose-adapter path，但为了顶刊，建议至少加入一个 non-RADIO backbone baseline，例如 DINOv2 或 SigLIP 2。否则“foundation model feature selection”会显得只是 RADIO-specific engineering。

---

## Layer B：Hypothesis Scorer

候选 (h_i) 可以有三种形式：

### B1. Reference image / reference pose candidate

不需要 3D render，最容易扩展到 Cambridge 多场景。

```text
query image feature vs reference image feature
→ candidate pose score
```

这一步最适合做 public multi-scene evidence。

### B2. Rendered candidate pose

使用 DCFF / 3DGS / feature field render candidate feature。

```text
query selected feature vs rendered map selected feature
→ candidate pose score
```

这对应你当前 OldHospital q10/q25/q50 positive evidence。

### B3. Real-init candidate set

来自：

```text
HLoc top-K
NetVLAD top-K
reference pose neighborhoods
perturbation around initializer
```

用来做 stress test，不作为主方法训练标签。

---

## Layer C：Optional handoff solver

这个不是主贡献。

如果要验证实际定位价值，固定使用：

```text
PnP-RANSAC + optional LM
```

或者外部 HLoc / MASt3R / GS-CPR-style pipeline 作为 baseline。

POFD-FS 只负责：

```text
rerank Top-K
select candidate for solver
predict risk / failure
```

不负责替代 MASt3R / LoFTR 做高精度 correspondence。

---

# 6. 训练目标应该改成这样

不要再把 teacher reconstruction 当主目标。

主损失：

## 6.1 Pose-distance listwise ranking

候选真实几何代价：

[
c_i = d(h_i, h^*)
]

soft target：

[
q_i = \mathrm{softmax}(-c_i / \tau_c)
]

模型预测：

[
p_i = \mathrm{softmax}(s_i / \tau_s)
]

损失：

[
L_{\text{rank}} = KL(q || p)
]

这比 one-hot classification 更适合 pose candidate，因为候选之间有连续几何结构。

---

## 6.2 Basin-aware classification

定义：

```text
positive: candidate inside solver basin
negative: candidate outside basin
```

例如：

```text
q10/q25: 10cm/5deg basin
q50: 25cm/10deg basin
```

损失：

[
L_{\text{basin}} = BCE(\hat{y}_i, y_i)
]

这个直接支撑论文 claim：

> selected features identify geometrically useful hypotheses.

---

## 6.3 Online hard negative loss

对每个 query：

```text
positive = oracle / basin-positive candidate
negative = 当前模型最高分但 pose cost 明显更差的 candidate
```

损失：

[
L_{\text{hard}} = \mathrm{softplus}(s_{\text{neg}} - s_{\text{pos}} + m)
]

这个是 q50 最需要的，因为当前 q50 仍有约 28.1% wrong top1 selections，错误选择的 mean cost 很高，说明剩余瓶颈就是 score-dominant hard wrong candidate。([GitHub][1])

---

## 6.4 Selection regularization

```text
channel sparsity
spatial utility entropy
low-rank projection regularization
counterfactual channel dropout
```

这里一定要做成可解释的 feature selection，而不是普通 descriptor training。

---

## 6.5 Mapability loss

对于多视角观察到的同一个 3D primitive / track：

[
L_{\text{map}} = Var{z_i(X)}
]

目标是证明：

> selected feature 不只是 2D ranking 好，也可以稳定聚合到 3D map。

---

# 7. 实验设计：顶刊最需要的证据链

## 7.1 主表 1：Foundation Feature Localizability

比较：

```text
raw RADIO
query student
RADIO PCA / random projection
teacher reconstruction
confidence branch
Fisher/logdet diagnostic
POFD-FS selected feature
POFD-FS selected feature + pair matcher scorer
```

指标：

```text
selected cost
oracle cost
oracle gap
top1
top5
Spearman
Kendall
NDCG
basin recall@K
hard-negative false accept rate
```

当前 OldHospital controlled ranking 里已经有雏形：

| feature / scorer                | q10 pred | q25 pred | q50 pred | q50 top1 | q50 Spearman |
| ------------------------------- | -------: | -------: | -------: | -------: | -----------: |
| raw RADIO + local r4            |    0.086 |    0.189 |    0.300 |    0.508 |        0.664 |
| query student + local r4        |    0.059 |    0.140 |    0.274 |    0.594 |        0.714 |
| pose-adapted + pair matcher r16 |    0.045 |    0.107 |    0.223 |    0.719 |        0.586 |

这个表可以成为论文核心，但 q50 还没过 promote threshold。([GitHub][1])

---

## 7.2 主表 2：定位范式分层

把同一个 feature 分别用于：

```text
place retrieval
hypothesis ranking
correspondence matching
continuous refinement
```

然后证明：

```text
foundation features 最强的是 hypothesis ranking；
direct continuous refinement 不稳定；
direct correspondence/PnP 不是最佳定位范式。
```

这张表会让你的论文比普通 ablation 更有理论深度。

---

## 7.3 主表 3：Second-scene / public-scene generality

这是当前最大短板。

新文档里明确说：ShopFacade controlled probes 还不是正结果，q50 coverage 不足，q25 top64 前 16 张 top1 仍为 0，因此 second-scene requirement 未满足。([GitHub][1])

为了最容易中稿，我建议不要只等 ShopFacade DCFF map 修好，而是加一个更容易扩展的 **reference-pose candidate ranking protocol**：

```text
Cambridge 5 scenes:
query image
candidate reference images / reference poses
feature scorer rerank candidates
label = candidate pose distance to GT
```

这样你可以利用已完成的 HLoc Cambridge 五场景基础设施。HLoc 已经在五个 Cambridge scenes 上跑完，并导出项目 pose-init caches；例如 ShopFacade `0.042m / 0.206deg`，OldHospital `0.144m / 0.309deg`，KingsCollege `0.114m / 0.210deg` 等。([GitHub][2])

这个 protocol 不需要每个场景都有 DCFF map，因此更容易形成 public multi-scene evidence。

---

## 7.4 主表 4：Mapability

指标：

```text
track-level feature variance
3D primitive feature variance
render-query residual
rendered candidate ranking
map storage dimension
risk calibration
```

比较：

```text
raw RADIO
query student
teacher reconstruction
POFD-FS selected
POFD-FS + mapability regularization
```

目标：

```text
selected feature 方差更低；
维度更低；
渲染后 pose ranking 不退化。
```

没有 mapability 证据，论文容易被质疑：

> 你只是做了 2D candidate scorer，不是 visual localization feature for maps。

---

## 7.5 主表 5：Real-init stress，不主打 SOTA

使用：

```text
HLoc
NetVLAD
external LoFTR / MASt3R baseline
```

只报告：

```text
Top-K basin recall
risk-coverage
catastrophic false accept rate
candidate reranking gain
failure prediction
```

不要把它写成 SOTA CPR 表。

当前 HLoc real-init 很强，render-at-init LoFTR 并不能稳定改善，POFD Stage4 也是 no-op；这说明 real-init final pose error 不是当前最容易赢的方向。([GitHub][2])

---

# 8. 最容易被接受的论文结构

## Section 1：Motivation

核心论点：

```text
Foundation features are strong for recognition and retrieval,
but localization requires different utilities.
```

已有 VPR 工作证明 foundation features 可用于 place recognition；SALAD 甚至引入 dustbin 丢弃 non-informative local features。([anyloc.github.io][3])

但本文要问：

```text
Which foundation feature subspaces are useful for 3D visual localization?
```

---

## Section 2：Taxonomy

定义四级 localizability：

```text
place
hypothesis
correspondence
differential
```

这是论文理论核心。

---

## Section 3：Benchmark

提出：

```text
Foundation Feature Localizability Benchmark
```

包含：

```text
controlled q10/q25/q50
reference-pose candidates
hard negatives
3D rendered candidates
real-init stress
```

---

## Section 4：Method

介绍 POFD-FS：

```text
feature selector
hypothesis scorer
basin-aware ranking loss
hard negative mining
mapability regularization
```

---

## Section 5：Experiments

按五张主表组织：

```text
feature localizability
paradigm decomposition
multi-scene generality
mapability
real-init stress
```

---

## Section 6：Findings

重点写：

```text
1. raw foundation features are not enough;
2. teacher reconstruction is insufficient;
3. hypothesis ranking is the sweet spot;
4. continuous refinement is not reliable with current foundation features;
5. selected features are compact, interpretable, and mapable.
```

---

# 9. 现在应该放弃的假设

从最容易中稿角度，我建议明确放弃这些：

## 放弃 1：必须达到 CPR SOTA

这会把你推向 GS-CPR / MASt3R 正面对比，而当前仓库证据不支持。GS-CPR 已经是“3DGS render + MASt3R correspondence + CPR”的强系统路线。([arXiv][4])

## 放弃 2：RADIO 特征必须直接 sparse correspondence + PnP

这不是 foundation feature selection 最稳的定位范式。

## 放弃 3：continuous feature-metric refinement 是主方法

当前 Stage4 no-go 可以作为 finding，不应继续当主线。

## 放弃 4：只做 OldHospital

顶刊不接受单场景故事。必须至少有一个 public multi-scene protocol。

## 放弃 5：只研究 RADIO

为了让题目叫 foundation model feature selection，至少加 DINOv2 或 SigLIP 2 对照。RADIO 可以是主 backbone，但不能是唯一证据。

---

# 10. 当前仓库下一步最关键的三件事

## 第一：把 q50 推过 promote threshold

当前 q50：

```text
pred = 0.223
top1 = 0.719
gap = 0.093
spearman = 0.586
```

目标：

```text
pred <= 0.21
top1 >= 0.75
gap <= 0.09
spearman >= 0.55
```

Spearman 已过，主要问题是 36/128 wrong top1 selections。文档已经指出，这不是 candidate index leakage，而是真实 q50 score-ordering failure。([GitHub][1])

下一步不要再扫 radius，也不要 naive selector projection。应该做：

```text
failure-conditioned hard negative training
```

具体是：

1. 从 rows.jsonl 中提取 36 个 wrong top1 query。
2. 按错误类型聚类：

   * selected outside basin
   * selected near repeated structure
   * oracle and selected both high score
   * selected score margin large
3. 构建 hard-negative replay buffer。
4. 训练时每个 batch 强制包含这些 failure rows。
5. 用 pairwise margin 直接压低 selected wrong candidate。

---

## 第二：做 public reference-pose ranking，不等 DCFF map

为了第二场景和 public evidence，最快路径是：

```text
Cambridge HLoc / reference-pose candidate ranking
```

输入：

```text
query image
top-K reference images / reference poses
foundation feature scorer
```

标签：

```text
pose distance to GT
basin label
```

指标：

```text
top1
top5
NDCG
basin recall@K
oracle gap
```

这不需要 3D render，所以最快能扩到 Cambridge 五场景。

这一步是顶刊必要证据。

---

## 第三：补 mapability

用 OldHospital / ShopFacade 已有 DCFF/feature assets 做：

```text
track variance
rendered feature ranking
selected feature storage dimension
```

没有 mapability，论文会像“候选分类器”，不够 visual localization feature paper。

---

# 11. 投稿前最低过线标准

我建议设成：

## 必须满足

```text
1. OldHospital q10/q25 明确优于 raw RADIO / query student / reconstruction。
2. OldHospital q50 过 promote threshold：
   pred <= 0.21, top1 >= 0.75, gap <= 0.09, spearman >= 0.55。
3. 至少一个 public multi-scene reference-pose ranking protocol 有正结果。
4. 至少一个 3D mapability 实验有正结果。
5. continuous CPR no-go 被写成定位范式分析，而不是失败实验。
```

## 强投稿标准

```text
1. Cambridge 5 scenes reference-pose ranking 全部跑通。
2. 至少 3/5 scenes 上 POFD-FS 优于 raw RADIO / DINOv2 / query-student baseline。
3. q50 OldHospital 和至少一个 second scene 有 positive ranking gain。
4. selected feature 维度 <= 64D。
5. channel / spatial utility 有 counterfactual ablation 支撑。
6. real-init stress 中 risk-coverage 或 basin recall@K 优于 baseline。
```

---

# 12. 可以直接给 Codex 的新 goal

```text
Goal:
Convert the current radio branch into a top-journal-oriented feature-centric paper:
"Which Foundation Features Localize? POFD-FS for Hypothesis-Based Visual Localization."

Main direction:
Do not pursue SOTA continuous CPR. The main contribution is a benchmark and method for selecting localization-usable subspaces from frozen foundation features, optimized for basin-aware pose-hypothesis ranking.

Hard constraints:
- Do not promote Stage4 continuous CPR as the main method.
- Do not use MASt3R/LoFTR/GS-CPR as the POFD-FS method; they are external baselines or teachers only.
- Do not use retrieval_scores_candidates or external PnP/LoFTR cache scores as main training labels.
- Do not claim SOTA pose refinement unless POFD-FS itself improves real-init final pose.
- Keep RADIO / foundation backbone frozen.
- Keep DCFF geometry frozen for mapability validation.
- Focus on hypothesis-localizability.

Tasks:

1. Q50 failure-conditioned hard-negative training:
   - Load current OldHospital q50 val128 dump.
   - Identify wrong top1 rows.
   - Build a hard-negative replay buffer.
   - Train the adapted pair-matcher score path with explicit selected-wrong vs oracle/basin-positive pairwise margin.
   - Target:
     pred <= 0.21, top1 >= 0.75, oracle_gap <= 0.09, Spearman >= 0.55.

2. Public reference-pose localizability benchmark:
   - Use Cambridge HLoc / reference pose data.
   - Build candidate banks for five scenes:
     ShopFacade, OldHospital, KingsCollege, GreatCourt, StMarysChurch.
   - Candidate source:
     top-K reference images / reference poses from retrieval or HLoc neighborhoods.
   - Labels:
     pose distance to GT, basin membership.
   - Evaluate:
     raw RADIO, query student, pose-adapted feature, POFD-FS selector.
   - Metrics:
     top1, top5, NDCG, Spearman, oracle_gap, basin_recall@K.

3. Mapability validation:
   - Use available OldHospital / ShopFacade maps.
   - Aggregate selected features to 3D primitives or tracks.
   - Report:
     track variance, primitive variance, rendered feature ranking, storage dimension.
   - Compare raw RADIO, query student, POFD-FS.

4. Interpretability:
   - Add counterfactual channel-group removal.
   - Add spatial utility top/bottom masking.
   - Report:
     ranking drop when removing high-utility channels/regions vs low-utility ones.
   - Export visualizations.

5. Paper report:
   - Create docs/superpowers/plans/2026-05-XX-pofd-fs-topjournal-plan.md
   - Include:
     taxonomy, method, benchmark, tables, failure modes, go/no-go conditions.

Promotion:
- OldHospital q50 passes threshold.
- At least 3/5 Cambridge scenes show nontrivial reference-pose ranking improvement over raw RADIO/query baseline.
- Mapability shows selected feature is lower variance and <=64D.
- Interpretability ablations show selected channels/regions are causally useful.
```

---

# 13. 最终建议

如果以“最容易中稿顶刊”为目标，本项目应该设计成：

> **一篇关于 foundation dense features 定位可用性的 benchmark + method + analysis paper。**

它的主线是：

```text
Foundation features are not equally useful for all localization paradigms.
They are weak as direct continuous pose-refinement signals,
but useful as hypothesis-ranking evidence.
POFD-FS learns a compact, interpretable, mapable subspace for this role.
```

这比继续追求：

```text
RADIO feature → sparse correspondence → PnP → SOTA CPR
```

要稳得多，也更符合当前仓库已经得到的真实证据。

[1]: https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-18-pofd-fs-localizability-mainline.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-17-iclp-completion-audit.md "raw.githubusercontent.com"
[3]: https://anyloc.github.io/?utm_source=chatgpt.com "AnyLoc: Towards Universal Visual Place Recognition"
[4]: https://arxiv.org/abs/2408.11085?utm_source=chatgpt.com "GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting"
[5]: https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/feature_extract/localizability/selector.py "raw.githubusercontent.com"