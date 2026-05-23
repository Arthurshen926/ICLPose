你的担心是对的：**如果只是说“hypothesis ranking 变好了”，这个 claim 还不够严谨，也不一定说明模型真的学到了定位相关知识。** 它可能只是学到了候选分布偏置、候选采样规律、near-identity prior，甚至是某些缓存字段里的 shortcut。要让这个方向成立，必须把“预期目标”和“特征选择有效性”重新定义得更严格。

我建议本项目的预期目标不是“直接做 SOTA camera pose refinement”，而是：

> **从 frozen foundation model dense features 中选择一个紧凑、可解释、可嵌入先验地图的 localization-usable feature subspace，使其在难例先验地图定位中提供比 raw foundation features、teacher reconstruction、传统 retrieval order、纯几何/光度评分更可靠的定位证据。**

这个目标里最关键的词是：**localization evidence**。
也就是说，选出的特征不一定直接完成 PnP 或连续 refinement，但必须能在一个真实定位流程中提供可量化、可泛化、可解释、可传递到地图的定位证据。

---

## 1. 先明确：hypothesis ranking 可以是严谨任务，但当前版本还不够

从已有工作看，“多候选 → verification / reranking”不是一个随意设定，而是视觉定位里的成熟范式。InLoc 和后续 pose verification 工作就是先生成多个候选位姿，再用 appearance、geometry、semantics 等证据做验证；Taira 等人的 *Is This the Right Place?* 明确指出，在弱纹理和重复几何模式的室内环境中，候选位姿 verification 是一个被低估但很重要的步骤。([arXiv][1])

所以，**hypothesis ranking 不是低级任务，也不是退而求其次**。它的问题在于：如果定义不严格，它很容易变成“候选表分类器”。

要让它变成严谨的定位任务，必须满足四个条件：

1. **候选生成器固定且独立**：候选由 HLoc、NetVLAD、reference retrieval、pose lattice、rendered map 等固定流程产生，不能由训练好的 scorer 反过来影响候选分布。

2. **标签来自几何或 solver 成功，而不是人工偏好**：候选的 label 应该是 pose error、是否在 solver basin、是否能被 fixed solver 成功修正，而不是“candidate index”或“某个 cache score”。

3. **scorer 的输入必须主要来自 query-map evidence**：不能只靠候选扰动幅度、候选排序、near-identity prior、PNP inlier 等 metadata。metadata 可以作为单独 baseline，但不能支撑“特征选择”claim。

4. **必须跨候选生成器和场景泛化**：如果只在 OldHospital 的 q10/q25/q50 candidate bank 上有效，而在 Cambridge reference-pose、ShopFacade、real-init hard cases 上无效，那它最多是局部诊断，不是通用 localization feature selection。

所以我会把任务升级为：

> **Map-conditioned localization hypothesis verification**
> 给定 query、先验地图和一组候选假设，选出的 foundation feature 必须提供可泛化的 query-map evidence，使正确或可修正的候选被排到前面。

---

## 2. 本项目的预期目标应该分三层

### 目标 A：科学目标

证明或证伪这个命题：

> **通用 foundation dense features 不是天然的精细定位特征，但其中存在一个可选择/蒸馏的子空间，能在先验地图定位的 hard hypotheses verification 中提供稳定定位证据。**

这个目标是合理的。DINOv2 一类模型被定位为 all-purpose visual features，RADIO/AM-RADIO 通过多教师蒸馏融合 CLIP、DINOv2、SAM 等不同视觉能力，而 SALAD 等 VPR 工作也说明 foundation features 可以通过聚合和丢弃 non-informative local features 服务地点识别。([arXiv][2])

但这些工作没有系统回答：

> 哪些 dense foundation features 对先验地图定位中的候选验证、hard negative 排除、map feature reconstruction 真正有用？

这才是本项目最值得做的科学问题。

---

### 目标 B：方法目标

提出一个完整方法，而不只是一个 scorer：

```text
Frozen foundation features
→ explicit feature selector / distiller
→ compact localization feature + utility + uncertainty
→ map-conditioned hypothesis scorer
→ selected feature lifting / mapability validation
→ optional solver handoff
```

也就是说，最终方法至少要包含三件事：

1. **选择**：从 foundation feature 中选择 channel / layer / spatial evidence。
2. **验证**：用定位任务证明这些 selected features 对候选判断有效。
3. **建图**：证明 selected features 可以被嵌入先验地图，并让 query 走同样的定位流程。

如果只有 1 和 2，没有 3，它更像 2D reranker。
如果只有 2，没有 1，它更像黑箱 scorer。
如果只有 3，没有 2，它只是 feature reconstruction。

---

### 目标 C：应用目标

解决已有定位方法容易失败的 hard cases，而不是全面宣称 SOTA。

这些 hard cases 应该明确定义为：

```text
1. retrieval top1 错，但 topK 内有正确区域；
2. repeated corridors / similar rooms / similar facades；
3. photometric rendering 看起来相似，但几何位置错误；
4. PnP / LoFTR / photometric refinement 在多个候选之间选择错误；
5. strong initializer 已经很接近，但存在 false positive refinement risk。
```

在这些场景上，选出的特征应该做到：

```text
减少 hard false accepts；
提高正确候选进入 topK 的概率；
提高 risk / failure prediction；
在固定 solver handoff 下提高成功率，或者至少减少灾难性退化。
```

---

## 3. 如何定义“特征选择是有效的”？

我建议定义四个必要条件。缺一个，claim 都会变弱。

---

### 条件 1：Predictive utility

选出的特征必须在固定候选集合上提高定位相关指标。

候选集合固定为：

```text
H = G(I, M)
```

其中 `G` 是固定候选生成器，例如：

```text
reference retrieval topK
HLoc topK
pose perturbation lattice
rendered pose hypotheses
photometric / PnP produced hypotheses
```

特征选择器输出：

```text
z = Sθ(Φ(I))
```

scorer 输出：

```text
s_i = score(z_q, z_map, h_i)
```

有效性指标不能只看 CE loss，而要看：

```text
top1 pose cost
topK oracle gap
Spearman / Kendall / NDCG
basin recall@K
hard-negative false accept rate
risk-coverage
```

当前仓库里已经有这个雏形。OldHospital controlled val128 上，raw RADIO + local r4 的 q50 pred 是 `0.300`、top1 `0.508`，query-student 提升到 `0.274`、top1 `0.594`，pose-adapted + pair matcher r16 提升到 `0.223`、top1 `0.719`；这说明 controlled hypothesis ranking 有正证据。([GitHub][3])

但这个正证据还不够，因为 q50 没稳定过 promotion threshold，public reference-pose ranking 还没有正结果，mapability 也没有证明。

---

### 条件 2：Causal selection

必须证明性能来自“选中的特征”，而不是 projection head、candidate prior 或 scorer shortcut。

需要做这些实验：

```text
1. raw full foundation feature；
2. same-dimensional random projection；
3. PCA projection；
4. teacher reconstruction feature；
5. full feature without sparse selection；
6. selected feature；
7. selected feature with high-utility channels removed；
8. selected feature with low-utility channels removed；
9. spatial high-utility regions removed；
10. spatial low-utility regions removed。
```

如果去掉 high-utility channel / region 后 ranking 明显变差，而去掉 low-utility channel / region 基本不变，才能说“选择”有因果意义。

当前仓库已经实现了 counterfactual diagnostics 接口，但文档也写到它还需要 exported selected-feature score maps 和 utility maps 才能成为论文结果。([GitHub][3])

所以目前只能说：

> 有选择模块和诊断接口，但还没有完成可投稿级“选择因果性”证据。

---

### 条件 3：Mapability

如果目标是“基于先验地图的定位”，selected feature 必须能进入地图。

也就是说，必须证明：

```text
selected 2D feature
→ 多视角聚合
→ 3D feature map
→ 渲染 selected map feature
→ query 使用同一 selected feature 流程定位
```

有效指标包括：

```text
track-level feature variance
primitive-level feature variance
rendered feature ranking retention
map storage dimension
render-query residual
uncertainty calibration
```

当前仓库已经有 raw RADIO mapability baseline，但文档明确说这还没有证明 selected POFD-FS mapability，因为 selected/query-adaptive dense caches 尚未导出。([GitHub][3])

这意味着：**当前还不能 claim selected feature 已经适合 3D map reconstruction。**

这点很关键，因为 GSFF 一类工作已经在做 3DGS feature field 和 2D encoder 的共同 embedding；新近工作 SplitGS-Loc 还指出 photometric GSFF 用于 2D-3D matching 时会有 many-to-one pixel-to-point ambiguity 和 multi-view consistency 问题。([arXiv][4])
因此，本项目如果要做“先验地图定位特征”，mapability 不是可选项，而是核心证据之一。

---

### 条件 4：Downstream utility

最终必须回答：

> 选出来的特征进入真实定位 pipeline 后，到底帮了什么？

不一定必须直接提高最终 pose SOTA，但至少要在下游任务中有一种明确价值：

```text
1. 提高 topK 中 solver-success candidate 的概率；
2. 降低 hard false positives；
3. 提高 accept/reject 的 risk calibration；
4. 减少已有方法在重复结构/弱纹理场景中的灾难性错误；
5. 在固定 solver handoff 下提升 final pose 或成功率。
```

当前 handoff 结果是不够的。文档里 q50 val128 的 POFD top1 再接 render-at-init RGB LoFTR+PnP 后 median 从 125mm 退化到 146mm；即使 oracle top8 diagnostic 也被该 solver 降级。因此目前只能说 solver-free ranking 有正证据，solver-conditioned localization 尚未成立。([GitHub][3])

这意味着：

> 现阶段不能 claim “selected feature improves final localization”。
> 可以 claim 的仍然是 “selected feature improves controlled hypothesis localization evidence”。

---

## 4. 你说“ranking 很难真正学到定位知识”，这个担心是合理的

尤其在当前 q50 结果中，这个问题已经暴露出来。

文档里 q50 failure clustering 显示：

```text
train q50 wrong top1: 55/96
train near-identity wrong: 5
val q50 wrong top1: 36/128
val near-identity wrong: 26
```

这说明验证集失败大量来自 near-identity false positives，而训练集没有同等分布。文档也明确说当前 q50 gap 更像 candidate-bank coverage problem，而不是 scalar calibrator capacity problem。([GitHub][3])

这就是 ranking 任务容易不严谨的地方：

```text
如果训练候选分布不覆盖真实 hard negatives，
模型可能学到的是候选采样分布，而不是定位证据。
```

所以，要让 ranking 学到定位知识，需要五个防作弊机制。

---

### 机制 1：metadata-only baseline

任何 candidate metadata 都必须单独建 baseline：

```text
candidate rank
delta from init
candidate perturbation magnitude
score margin
PNP inlier count
photometric residual
```

如果 feature scorer 不能显著超过 metadata-only baseline，就不能说学到了 feature localization knowledge。

尤其当前 score calibrator 支持 `score_margin_to_top1`、`score_rank_norm`、`score_zscore`、`delta_trans_m`、`delta_rot_deg` 等 candidate evidence。文档已经提示 scalar candidate evidence 接近饱和，继续扩大 MLP 不是 clean fix。([GitHub][3])
因此，calibrator 结果不能作为强“feature selection”证据，只能作为 ranking diagnostic。

---

### 机制 2：feature-shuffle negative control

做三种 shuffle：

```text
1. query feature shuffled across queries；
2. candidate feature shuffled across candidates；
3. map feature from wrong scene；
```

如果 shuffle 后模型仍然有效，说明它没有真正用 query-map feature evidence。

---

### 机制 3：candidate generator transfer

训练在一种候选生成器上：

```text
q50 lattice
```

测试在另一种候选生成器上：

```text
HLoc reference-pose topK
NetVLAD topK
randomized q50 bank
hard near-identity bank
```

如果只在一种 bank 上有效，就是 bank-specific shortcut。

---

### 机制 4：scene transfer

至少需要：

```text
OldHospital → ShopFacade / Cambridge scene
```

当前 public multi-scene bank 已建好，但 raw RADIO pooled feature 和 query-student pooled descriptor 都比 retrieval-order baseline 差。文档明确说这只是 public substrate，还不是 POFD-FS positive result。([GitHub][3])

下一步必须有 learned patch-level / local-evidence scorer 在 public scene 上获得正结果。

---

### 机制 5：hard-case selection

不能只在平均样本上看 pred。要专门构建 hard set：

```text
retrieval top1 wrong but topK has correct；
HLoc close but has repeated-structure ambiguity；
photometric score prefers wrong candidate；
near-identity false positive；
similar room / corridor；
weak texture / low overlap。
```

选出的特征必须在 hard set 上显著减少 false accept。否则它没有解决“已有方法解决不了的难例”。

---

## 5. 本项目的最终评估框架应该是“三道门”

我建议把评估分成三道门。每一道门都对应一个 claim。

---

### Gate 1：Feature localizability

问题：

> 选出的 feature 是否比 raw foundation feature 更适合候选判断？

指标：

```text
top1 candidate cost
oracle gap
Spearman / NDCG
basin recall@K
hard-negative false accept
```

必须比较：

```text
raw RADIO
raw DINOv2 / optional SigLIP
same-dim random projection
PCA
teacher reconstruction
query-student
pose-adapted feature
explicit selector
metadata-only scorer
```

当前 OldHospital q10/q25 支持这个 gate，q50 接近但还不稳。([GitHub][3])

---

### Gate 2：Feature selection causality + mapability

问题：

> 它真的选择了有用特征吗？这些特征能进地图吗？

指标：

```text
channel counterfactual drop
spatial counterfactual drop
selected dimension
track variance
rendered ranking retention
map storage cost
uncertainty calibration
```

当前接口有，但 selected mapability 证据还缺。([GitHub][3])

---

### Gate 3：Downstream hard-case utility

问题：

> 它是否帮助现有定位系统解决难例？

可选任务：

```text
reference-pose reranking
pose-hypothesis verification
solver accept/reject
topK handoff selection
photometric / PnP failure prediction
```

指标：

```text
hard-case success rate
catastrophic false accept reduction
risk-coverage AUC
solver-conditioned success@K
final pose error if solver is stable
```

当前 Gate 3 没成立，因为 existing PnP/LoFTR quality fields 和 naive LoFTR handoff 都不可靠。([GitHub][3])

---

## 6. 本项目的预期目标应该具体设成什么？

我建议分成最低目标、强目标、放弃条件。

---

### 最低可投稿目标

如果要写成 feature-centric paper，最低应达到：

```text
1. OldHospital controlled:
   q10/q25 selected feature 明显优于 raw RADIO、query-student、teacher reconstruction；
   q50 至少稳定达到 pred <= 0.21、top1 >= 0.75、gap <= 0.09、Spearman >= 0.55。

2. 至少一个第二场景:
   ShopFacade 或 Cambridge reference-pose ranking 上，
   learned selected feature/scorer 优于 raw RADIO 和 retrieval-order baseline 中至少一个强基线。

3. Causal selection:
   channel/spatial counterfactual ablation 证明高 utility 特征确实更重要。

4. Mapability:
   selected feature 的 track variance 或 rendered ranking retention 优于 raw/reconstruction；
   feature dimension <= 64。

5. Hard-case utility:
   在人工定义的 hard set 上减少 false accept 或提高 topK basin recall。
```

这能支撑：

> 我们提出并验证了一种 foundation feature selection 方法，用于 prior-map hypothesis localization。

---

### 强投稿目标

如果目标是 TRO/顶刊级别，应该达到：

```text
1. 至少 3 个场景，其中至少 1 个 public dataset。
2. 至少 2 种 hypothesis type：
   reference-pose candidate + rendered pose candidate。
3. 至少 2 种 candidate generator：
   retrieval/HLoc + randomized pose lattice。
4. selected feature 在多场景上都优于 raw foundation features、teacher reconstruction、metadata-only scorer。
5. selected feature 可嵌入 3D map，并且用 reconstructed selected map feature 跑同一套 scorer 仍有效。
6. 下游 fixed solver 或 verification protocol 上有稳定 hard-case improvement。
```

这才能支撑更强 claim：

> 选出的 feature 不只是 OldHospital q50 bank 上有效，而是真正的 map-conditioned localization evidence。

---

### 放弃或降级条件

如果出现以下情况，应降级成 diagnostic / benchmark paper，不再 claim 方法强有效：

```text
1. q50 必须依赖 metadata calibrator 或 candidate prior 才能提升；
2. learned feature 不能在第二场景优于 raw/retrieval baseline；
3. selected feature mapability 差，进入 3D 后 ranking 消失；
4. feature shuffle 后性能不明显下降；
5. hard-case false accept 没有减少；
6. solver-conditioned protocol 完全没有收益。
```

---

## 7. 我建议把“hypothesis ranking”改成更严谨的名字

不要叫普通 hypothesis ranking。建议叫：

> **Localization Evidence Selection for Map-Conditioned Hypothesis Verification**

简称可以是：

```text
LES-HV
```

或者继续叫：

```text
POFD-FS: Pose-Observable Foundation Feature Selection
```

但论文里要强调它不是一般排序，而是：

```text
fixed hypothesis generator
+ query-map feature evidence
+ basin-aware label
+ hard-negative verification
+ mapability constraint
+ downstream handoff validation
```

这样才严谨。

---

## 8. 最终回答你的问题

### “本项目的预期目标是什么？”

不是“用 foundation feature 直接做精细定位”，也不是“单纯做候选表排序”。

更准确的目标是：

> **学习一个可解释、紧凑、可建图的 foundation feature 子空间，使其在先验地图定位中的 hard hypotheses verification 任务上，提供比 raw foundation features、teacher reconstruction、retrieval order、metadata-only prior 更可靠的定位证据，并能在 selected 3D feature map 中复用。**

---

### “如何评估特征选择是有效的？”

至少要同时满足：

```text
1. 性能有效：
   ranking / basin / hard-negative metrics 明显提升。

2. 因果有效：
   去掉 selected high-utility 特征会显著变差；
   random/PCA/teacher reconstruction 打不过 selected feature。

3. 泛化有效：
   换场景、换 candidate generator、换 hard negatives 仍有效。

4. 地图有效：
   selected feature 能聚合到 3D 地图，渲染后仍可用。

5. 下游有效：
   至少在 hard-case verification、risk prediction、topK solver handoff 中有实际收益。
```

---

### “hypotheses ranking 感觉不够严谨，难学到定位知识？”

你这个担心非常合理。当前版本的 ranking 还不够严谨，尤其 q50 结果已经显示训练候选分布和验证 hard negatives 不一致。要让它学到定位知识，必须加入：

```text
metadata-only baseline
feature-shuffle control
candidate-generator transfer
scene transfer
solver-measured basin labels
hard-case false-accept evaluation
mapability validation
```

否则它确实可能只是学到候选表规律。

所以，本项目下一步不应该继续简单调 ranking loss，而应该把 ranking 升级成：

> **受控、反泄漏、跨场景、map-conditioned 的 localization hypothesis verification benchmark。**

只有这样，特征选择这个任务才逻辑完备，也才可能支撑顶刊 claim。

[1]: https://arxiv.org/abs/1908.04598?utm_source=chatgpt.com "Is This The Right Place? Geometric-Semantic Pose Verification for Indoor Visual Localization"
[2]: https://arxiv.org/abs/2304.07193?utm_source=chatgpt.com "DINOv2: Learning Robust Visual Features without Supervision"
[3]: https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-21-pofd-fs-topjournal-plan.md "raw.githubusercontent.com"
[4]: https://arxiv.org/abs/2507.23569?utm_source=chatgpt.com "Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization"
