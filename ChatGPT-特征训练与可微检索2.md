# 特征训练与可微检索

**User:** Anonymous  
**Created:** 5/7/2026 20:37:27  
**Updated:** 5/10/2026 12:36:35  
**Exported:** 5/10/2026 12:54:05  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00-shen-qi-yuan/c/69fc8787-36c8-83ea-bcd2-c2221cf5eece](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00-shen-qi-yuan/c/69fc8787-36c8-83ea-bcd2-c2221cf5eece)  

## Prompt:
2026/5/10 11:48:26

目前按照我们讨论的内容推进了一下，具体情况请看https://github.com/Arthurshen926/ICLPose/tree/radio 最新推送，当前离目标的主要瓶颈仍是：coarse top1 不稳，topK 有用，fine-score rerank 有用，但 map-side candidate-gradient 还会把中 basin 拉坏。请具体帮我分析问题在哪里？该如何修改？

## Response:
2026/5/10 11:48:30

我会先重新核对最新 `radio` 分支的主线配置、训练/评估入口、coarse topK、fine rerank、candidate-gradient 和 stop-grad 边界。然后重点判断这些信号是否在监督目标、候选排序和 map-side 更新上互相打架。

## Response:
2026/5/10 11:48:41

我已经看到一个关键新信号：主线已转向 learned top5 init、full_wls 和 fine-score rerank，但仍有配置解冻 `finetune_decoder/finetune_fsm`。这说明核心不是 topK 无信息，而是如何用 topK 同时避免 map-side 被 candidate 梯度污染。

## Response:
2026/5/10 11:52:07

> **Clarifying code citation and raw retrieval options**
>
> [raw.githubusercontent.com](https://github.com/Arthurshen926/ICLPose/raw/refs/heads/radio/feature_retrieval/tools/train_candidate_selector.py)
>
> Thought for 10m 33s

## Response:
2026/5/10 11:59:03

我看了最新推送后的主线和阶段总结。我的判断很明确：

$$
\boxed{
\text{现在不要再把 map-side candidate-gradient 放进主线。}
}
$$

你的现象其实已经把问题说透了：**coarse top1 不稳，但 topK 有用；fine-score rerank 有用；map-side candidate-gradient 会把中 basin 拉坏。** 这说明系统不是“没有 coarse 信号”，而是 **coarse 分支适合作 recall generator，不适合现在被训练成 hard top1 selector；map-side candidate-gradient 的信用分配位置错了**。

换句话说，问题不是：

> map feature 还不够端到端。

而是：

> 你现在让 map feature 通过 candidate ranking loss 去承担一个离散、偏 top1、偏局部候选集的选择目标，结果它把原本连续的 refinement basin 破坏了。

这和之前专家简报里的判断一致：当前最弱段仍是 candidate scoring / selector / initialization，而不是后端 refiner 完全没有能力。`EXPERT_REVIEW_BRIEF_2026-05-07.md`

---

# 1. 现在真正的问题在哪里

## 1.1 coarse top1 不稳，不是 bug，而是目标定义不对

你当前 Phase5 文档里已经明确：当前主线是

$$
T_0 \rightarrow \text{local pose lattice} \rightarrow \text{frozen coarse candidate scorer topK} \rightarrow \text{fine-score rerank}
$$

并且 `fine_update_scale=0` 是刻意设置；fine branch 现在主要用于 **从 topK 里选候选**，不是强行做 WLS 位姿更新。文档也写得很直接：confidence-selected WLS 曾经让 pose 变差，所以当前 fine score 更像 selector，而不是 solver。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

这套结果其实很合理：

- coarse top1 不稳；
- coarse top4/top8 有效；
- fine-score rerank 可以恢复很多；
- WLS/confidence 还不能作为强更新主线。

所以你现在不应该逼 coarse top1 成为主系统入口。正确设计应该是：

$$
\boxed{
\text{coarse} = \text{topK recall / basin proposal}
}
$$

$$
\boxed{
\text{fine-score} = \text{topK rerank / recoverability selector}
}
$$

$$
\boxed{
\text{WLS update} = \text{diagnostic 或后续增强，不是当前主线}
}
$$

现在 coarse top1 不稳，最可能不是单纯实现 bug，而是 **loss 和任务职责不匹配**。coarse score 看到的是低分辨率、较鲁棒但模糊的 pose evidence；在 0.25m/5°、0.5m/10° 这种局部 lattice 里，很多候选的渲染特征差异本来就很小，要求它精确 top1 是过强要求。

---

## 1.2 topK 有用，说明 candidate coverage 不是当前第一瓶颈

你最新总结里已经有很关键的证据：在 frozen medium 下，coarse top1 median 约 353.5mm，但 top4 recall 约 0.625，top8 recall 约 0.844；Phase5 GT-align map 在 0.25m 和 0.5m bucket 上也有更好的 mean 结果。这说明候选集里常常已经包含可用 pose，只是排序和最终选择不稳。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

这意味着接下来不该优先问：

> 怎么让 coarse top1 直接正确？

而该问：

> 怎么让 coarse 保证 topK 覆盖，然后让 fine-score 稳定选出可 refine 的候选？

所以主指标也要改。不要再把 top1 accuracy 当主目标。主指标应该是：

$$
\text{topK basin recall}
$$

$$
\text{fine-rerank final mean error}
$$

$$
\text{gain+ rate}
$$

$$
\text{full-lattice mean error}
$$

尤其不要被 exact 0mm median 欺骗。你文档里也已经指出 exact 0mm median 是 fixed-noise artifact，应该看 mean、gain+、topK recall。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

---

## 1.3 fine-score rerank 有用，说明 fine feature 的当前最佳角色是 selector

现在 fine-score rerank 有用，说明 fine branch 已经提供了比 coarse score 更接近“这个候选能不能对齐 query”的信号。它不一定能稳定产生 WLS update，但它能评价 candidate 的 local alignment quality。

所以当前 pipeline 应该正式改成：

$$
\text{coarse topK proposal}
\rightarrow
\text{fine-score rerank}
\rightarrow
\text{selected pose}
$$

而不是：

$$
\text{coarse top1}
\rightarrow
\text{fine WLS update}
$$

也不是：

$$
\text{candidate-gradient}
\rightarrow
\text{map feature 变得更适合 top1}
$$

你的 FM-GN 总结也支持这个判断：FM-GN 的收敛 basin 很窄，translation 有时能 gain，但 rotation drift 明显，gradient direction loss 和 FM-GN supervision 都还没把它变成稳定的主力 solver。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-fm-gn-%E5%AE%9E%E9%AA%8C%E6%80%BB%E7%BB%93.md))

---

# 2. map-side candidate-gradient 为什么会拉坏中 basin

这是现在最关键的点。我认为原因有四个。

---

## 2.1 candidate-gradient 在优化“局部候选集排序”，不是优化连续 pose basin

你最新文档里已经记录了几个负结果：

- K32 uniform candidate-gradient 停掉，因为 validation oracle/top8 只有 0.375；
- mixed K49 因为 25cm/5° 和 50cm/10° 混合后 compact outer lattice 不保证严格 basin coverage；
- outer50 K49 在 step32 coverage 看起来干净，但 full medium eval 把 0.5m mean 从 167.3mm 拉到 264.7mm；
- oracle-subset candidate-gradient 可以帮助 0.25m，但会伤 0.5m。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

这说明 candidate-gradient 学到的是：

$$
\text{在当前小候选子集里把某些 candidate 分开}
$$

而不是：

$$
\text{构造一个对 0.25m 到 0.5m 都平滑有效的 pose energy landscape}
$$

所以它会出现典型现象：

$$
\text{small basin 变好，middle basin 变坏}
$$

这是因为 ranking loss 会把 map feature 拉向离散分类边界，而不是拉向连续几何对齐。

---

## 2.2 你现在解冻的 map-side 自由度太大

`cpr_main_phase5_map_finetune_lattice.yaml` 这类配置里，map supervision 是 trainable，并且会训练 fine decoder、coarse fusion、FSM、hash MLP、latent 等多个部分；同时还开了 candidate render score train map。这个不是“小范围 map-side candidate-gradient”，而是一个很大的 map feature 重参数化。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/feature_extract/configs/cpr_main_phase5_map_finetune_lattice.yaml))

这会造成一个问题：

> candidate loss 本来就很离散、很偏排序；你却给了它足够大的自由度去改整个 map feature field。

结果自然是：它可能在当前候选集上把 score 变好，但破坏 fine-score rerank 依赖的稳定 feature geometry。

对当前阶段来说，map-side candidate-gradient 不应该更新 core map field。最多只能更新一个很小的 **candidate adapter**。

---

## 2.3 candidate loss 权重相对 anchor 太强

你的 clean oracle-subset candidate-gradient 配置里，虽然已经更安全：query/scorer frozen、query detach、候选子集包含 GT-nearest + uniform negatives，但 candidate score fusion weight 仍然是 0.5，而 query fine/coarse anchor 只有 0.02，teacher anchor 约 0.0005。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/feature_extract/configs/cpr_clean_phase5_maponly_oraclesubset33_medium_b1.yaml))

这等于说：

$$
L_{\text{candidate-rank}}
\gg
L_{\text{query-map-align}} + L_{\text{teacher-anchor}}
$$

所以 map feature 被 ranking loss 拉着走，而 RADIO / query-map alignment / multi-view consistency 对它的约束太弱。

这会导致 map feature 变成：

$$
\text{candidate classifier feature}
$$

而不是：

$$
\text{pose refinement feature}
$$

这就是中 basin 被拉坏的核心原因之一。

---

## 2.4 scorer 和 map 同时动，会形成 moving target

在主 Phase5 配置里，candidate score fusion head 仍然是 trainable，同时 map side 也是 trainable。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/feature_extract/configs/cpr_main_phase5_map_finetune_lattice.yaml))

这很危险。因为 scorer 在变，map feature 也在变。你得到的是：

$$
S_{\eta_t}(R_{\phi_t}(\mathcal{M}, T))
$$

其中 scorer 参数 $\eta_t$ 和 map 参数 $\phi_t$ 同时移动。这样 candidate loss 的信用分配非常不稳定：到底是 scorer 排错了，还是 map feature 错了，训练过程分不清。

当前阶段应该遵守一个硬规则：

$$
\boxed{
\text{训练 scorer 时 freeze map；训练 map 时 freeze scorer。}
}
$$

不要同时训。

---

# 3. 我认为当前没有必要继续追 coarse top1

现在应该把论文/系统主线改成：

$$
\boxed{
\text{coarse topK proposal + fine-score rerank}
}
$$

而不是：

$$
\boxed{
\text{coarse top1 localization}
}
$$

具体说：

| 模块 | 当前应该负责 | 不应该负责 |
|---|---|---|
| coarse scorer | high recall topK | 直接 top1 最终定位 |
| fine-score | topK 内 recoverability rerank | 强 WLS update |
| WLS / FM-GN | 诊断、可选 refinement | 当前主线强更新 |
| map-side finetune | GT-align / feature stability | candidate ranking 主优化 |
| candidate-gradient | 小 adapter 探针 | core map field 更新 |

这个和仓库最新 Phase5 文档里的“当前默认 report path：topK=8、fine_select=fine_score、fine_update_scale=0”是一致的。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

---

# 4. 具体怎么改

## 4.1 立刻把主线锁成 frozen / safe-map 路径

当前主线建议：

```text
topK = 8
fine_select = fine_score
fine_update_scale = 0
map-side candidate-gradient = off
```

论文/实验主表先报这个。

也就是说，主系统变成：

$$
T_0
\rightarrow
\text{coarse local lattice}
\rightarrow
\text{top8}
\rightarrow
\text{fine-score rerank}
\rightarrow
T^*
$$

先不要在主表里用 candidate-gradient map finetune。Phase5 文档也已经建议：保留 Phase5 GT-align，不把 candidate-gradient 作为 mainline。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

---

## 4.2 coarse scorer 的 loss 改成 topK / multi-positive，不要 top1 CE

现在 coarse top1 不稳，所以 loss 也要反映它的职责。

不要用：

$$
L = -\log p_{k^+}
$$

这种强 top1 loss 作为主目标。

改成 multi-positive InfoNCE / topK recall loss：

$$
P(q) =
\{k \mid d(T_k, T_{gt}) < \delta\}
$$

$$
L_{\text{topK}} =
-\log
\frac{
\sum_{k \in P(q)} \exp(s_k / \tau)
}{
\sum_j \exp(s_j / \tau)
}
$$

其中 positive 不应该只有 GT-nearest。应该包括：

1. pose error 低于阈值的候选；
2. fine-score rerank 后表现好的候选；
3. refiner 可恢复的候选。

更好的 target 是 refine-aware：

$$
k^+ =
\arg\min_k d(T^{\text{fine-select}}_k, T_{gt})
$$

而不是：

$$
k^+ =
\arg\min_k d(T_k, T_{gt})
$$

因为你的系统最终不是 coarse-only，而是 coarse + fine rerank。

---

## 4.3 candidate target 要从 raw pose error 改成 recoverability

当前 Phase4 frozen map config 里 candidate score target 是 `gt_pose_error_soft`，这适合做第一版，但它不完全符合系统实际部署路径。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/feature_extract/configs/cpr_main_phase4_coarse_lattice_frozenmap.yaml))

因为有些候选 pose 不是 GT-nearest，但 fine-score 更容易选对，或者局部 alignment 更可靠。对你的系统来说，candidate 的好坏应该定义成：

$$
\text{good candidate}
=
\text{fine-score 能选中且 final error 小}
$$

所以建议 target cost 改成：

$$
C_i =
\alpha d_{\text{pose}}(T_i, T_{gt})
+
\beta C_{\text{fine-align}}(T_i)
+
\gamma C_{\text{recover}}(T_i)
$$

实践里可以先简单一点：

```text
positive candidates =
  pose-nearest top 1-3
  +
  fine-score selected best
  +
  candidates within 0.25m/5° or 0.5m/10°
```

negative 用：

```text
hard negatives =
  high coarse score but high pose error
  +
  high fine score but wrong pose
  +
  boundary negatives around basin threshold
```

不要主要用 uniform negatives。uniform negatives 太容易，学不到 selector 的关键边界。

---

## 4.4 map-side candidate-gradient 只能作为 adapter，不要更新 core map

这是我最强的修改建议。

现在 candidate-gradient 会拉坏中 basin，本质原因是它在改 core map feature。你应该加一个很小的 candidate-only adapter：

$$
F^c_{\text{score}}
=
\text{norm}
\left(
F^c_{\text{map}}
+
\epsilon A_\psi(F^c_{\text{map}})
\right)
$$

其中：

- $F^c_{\text{map}}$：原始 coarse map feature，冻结；
- $A_\psi$：小 MLP / 1x1 conv / low-rank adapter；
- $\epsilon$：小系数，比如 0.1；
- candidate-gradient 只更新 $A_\psi$。

fine-score rerank 和 fine feature 不走这个 adapter。这样 candidate-gradient 即使过拟合，也不会破坏 fine branch 和中 basin feature geometry。

也就是说：

```text
base map feature: frozen or GT-align only
candidate scoring feature: base feature + tiny adapter
fine rerank feature: base fine feature
```

这比直接训练 hash MLP、latent、FSM、decoder 安全得多。

---

## 4.5 如果一定要 map-side candidate-gradient，只能这样做

候选选择必须是 no-grad full lattice，render-grad 只能走小子集。

伪代码应该是：

```python
# 1. full lattice no-grad scoring
with torch.no_grad():
    all_candidates = build_full_lattice(T0)          # e.g. K169
    all_feats = render_map_no_grad(all_candidates)
    all_scores = frozen_scorer(query_feat, all_feats)
    all_pose_cost = pose_cost(all_candidates, T_gt)

    pos_idx = select_positives(
        all_pose_cost,
        fine_score_cache,
        thresholds=[0.25, 0.5],
    )

    hard_neg_idx = select_hard_negatives(
        all_scores,
        all_pose_cost,
        min_pose_gap=True,
    )

    selected_idx = stratified_sample(pos_idx, hard_neg_idx)

# 2. only selected subset uses grad render
selected_candidates = all_candidates[selected_idx]
rendered_feats = render_map_with_grad(selected_candidates)

# 3. scorer frozen, query detached
logits = frozen_scorer(query_feat.detach(), rendered_feats)

loss_rank = listwise_rank_loss(logits, target[selected_idx])
loss_anchor = gt_align_loss(...) + teacher_loss(...) + drift_loss(...)

loss = loss_anchor + small_lambda * loss_rank
```

关键是：

```text
full lattice 负责选样本和算 target
small subset 负责省显存和回传梯度
```

不能让 compact K33/K49 本身决定训练世界。你文档里的 mixed K49 / outer50 / oracle-subset 结果已经证明：候选子集一旦和 full-lattice 分布不一致，map-side gradient 很容易伤中 basin。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

---

## 4.6 candidate rank loss 权重要降一个数量级

现在 candidate score fusion weight 用到 0.5 太激进。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/feature_extract/configs/cpr_clean_phase5_maponly_oraclesubset33_medium_b1.yaml))

我建议：

```yaml
candidate_score_fusion_weight: 0.02   # 先从 0.02 或 0.03 开始
candidate_score_fusion_pairwise_rank_weight: 0.02
```

同时 anchor 加强：

```yaml
query_fine_weight: 0.10
query_coarse_weight: 0.05
rendered_teacher_fine_weight: 0.002
rendered_teacher_coarse_weight: 0.002
map_feature_drift_weight: 0.10
```

也就是：

$$
L =
L_{\text{GT-align}}
+
L_{\text{teacher}}
+
L_{\text{drift}}
+
0.02 L_{\text{candidate-rank}}
$$

不要再让：

$$
L_{\text{candidate-rank}}
$$

支配 map feature。

---

## 4.7 加 old-map drift regularization

你需要显式防止 map feature 被 candidate loss 拉偏。做一个 frozen old map：

$$
F_{\text{old}}(T) = R_{\text{old}}(\mathcal{M}, T)
$$

训练新 map 时加：

$$
L_{\text{drift}}
=
1 -
\cos
\left(
F_{\text{new}}(T),
F_{\text{old}}(T)
\right)
$$

采样 pose 不要只用 positive candidate，也要覆盖：

```text
GT pose
0.25m/5° candidates
0.5m/10° candidates
hard negatives
random nearby poses
```

这个 regularizer 的意义是：

> candidate-gradient 可以轻微塑形 score adapter，但不能破坏原来的连续 feature field。

---

## 4.8 不要同时训练 scorer 和 map

改训练阶段：

### Stage A：frozen map，训练 scorer / reranker

```yaml
map_supervision:
  trainable: false

train:
  candidate_score_fusion_head: true
  fine_score_reranker: true
```

目标：

$$
\text{topK recall}
+
\text{fine-rerank final mean}
$$

### Stage B：frozen scorer，GT-align map

```yaml
map_supervision:
  trainable: true
  candidate_render_score_train_map: false
  train_fine_decoder: true
  train_coarse_fusion: false
  train_hash_mlp: false
  train_latent: false
  train_fsm: false
```

这一步只做 safe map alignment。你的 Phase5 GT-align 结果已经说明这条路更稳。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/feature_extract/configs/cpr_clean_phase5_maponly_gtalign_medium_b4.yaml))

### Stage C：frozen scorer，candidate adapter probe

```yaml
map_supervision:
  trainable: true
  candidate_render_score_train_map: true
  train_candidate_score_adapter: true

  train_fine_decoder: false
  train_coarse_fusion: false
  train_hash_mlp: false
  train_latent: false
  train_fsm: false
```

这一步只作为 ablation，不要作为主线。

---

# 5. 推荐你马上改的配置

## 5.1 safe-map 主线配置

把 Phase5 GT-align 变成主线，而不是 candidate-gradient 主线。

建议类似：

```yaml
training:
  lr: 2.0e-5
  best_metric: full_lattice_top8_fine_rerank_mean_mm
  best_metric_mode: min

model:
  fine_update_scale: 0.0

map_supervision:
  trainable: true
  detach_query_features: true

  candidate_render_score_train_map: false
  candidate_score_fusion_weight: 0.0

  train_fine_decoder: true
  train_coarse_fusion: false
  train_hash_mlp: false
  train_latent: false
  train_fsm: false
  train_geometry: false
  train_color: false

  query_fine_weight: 0.10
  query_coarse_weight: 0.05
  rendered_teacher_fine_weight: 0.002
  rendered_teacher_coarse_weight: 0.002
```

注意：先不要训练 hash MLP / latent / FSM。那些自由度太大，会掩盖问题。

---

## 5.2 scorer 主线配置

frozen map 下训练 topK scorer：

```yaml
map_supervision:
  trainable: false

candidate_score_fusion:
  enabled: true
  target_mode: refine_aware_multi_positive
  target_temp: 0.25
  pairwise_rank_weight: 0.05

coarse_pose_lattice:
  use_full_lattice_for_target: true
  train_render_subset: true
  topk_metric: top8_basin_recall
```

目标不是 top1，而是：

```text
top8_basin_recall
fine_rerank_final_mean
gain+
```

---

## 5.3 candidate-gradient probe 配置

如果你还要继续试 map-side candidate-gradient，建议强制变成 adapter-only：

```yaml
map_supervision:
  trainable: true
  detach_query_features: true

  candidate_render_score_train_map: true
  candidate_score_fusion_weight: 0.02
  candidate_score_fusion_pairwise_rank_weight: 0.02

  train_candidate_score_adapter: true

  train_fine_decoder: false
  train_coarse_fusion: false
  train_hash_mlp: false
  train_latent: false
  train_fsm: false
  train_geometry: false
  train_color: false

  map_feature_drift_weight: 0.10
  query_fine_weight: 0.10
  query_coarse_weight: 0.05
  rendered_teacher_fine_weight: 0.002
  rendered_teacher_coarse_weight: 0.002
```

并且加硬 early stop：

```text
if top8_basin_recall@0.5m drops:
    stop

if fine_rerank_final_mean@0.5m worsens > 10%:
    stop

if gain+@0.5m drops:
    stop
```

Phase5 文档已经建议：如果 gain+ drops 或 full-lattice mean regresses，就停止 map finetune。这个判断是对的。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

---

# 6. 指标也要改，否则你会被错误指标带偏

当前不要再用 compact subset 的 `pred_trans` 或 exact-median 选 checkpoint。应该用 full lattice validation。

建议每次验证都报：

```text
coarse_top1_error
coarse_top4_basin_recall
coarse_top8_basin_recall
fine_rerank_final_mean
fine_rerank_gain+
selected_not_top1_rate
0.25m/5° bucket mean
0.5m/10° bucket mean
```

尤其要分 bucket：

```text
0.25m/5°
0.5m/10°
1.0m/20°
```

你现在最怕的是：

$$
0.25m \text{ 变好}
$$

但：

$$
0.5m \text{ 变坏}
$$

所以 best checkpoint 不能只看 small basin。

---

# 7. 我认为最可能的“代码级坑”

这几个地方我会优先检查。

## 7.1 candidate subset 是否真的包含 GT-nearest

你文档里已经说 K32 uniform candidate-gradient 污染了训练集，oracle/top8 只有 0.375。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

所以训练时必须 assert：

```python
assert gt_nearest_idx in selected_idx
```

或者至少：

```python
assert min_pose_cost(selected_candidates) < threshold
```

否则 candidate-gradient 训练的是错误排序。

---

## 7.2 full-lattice target 和 render-grad subset 是否一致

训练可以只 render 小 subset，但 target 必须来自 full lattice。否则 map 会学 compact subset 的偏置。

加日志：

```text
full_lattice_best_cost
selected_subset_best_cost
selected_contains_full_best
top8_contains_positive
```

如果 `selected_contains_full_best = false`，这个 batch 的 candidate-gradient 不应该更新 map。

---

## 7.3 scorer 是否在 map finetune 时仍然 trainable

这点很关键。

map candidate-gradient 时必须：

```python
for p in candidate_score_fusion_head.parameters():
    p.requires_grad_(False)
```

但 scorer forward 不能放在 `torch.no_grad()` 里，否则梯度也不能回到 rendered map feature。正确做法是：

```python
freeze scorer params
detach query features
allow gradient through rendered map features
```

即：

```python
query_feat = query_feat.detach()
logits = frozen_scorer(query_feat, rendered_map_feat_with_grad)
loss.backward()
```

---

## 7.4 best metric 是否和部署路径一致

如果当前部署路径是：

```text
topK + fine-score rerank + fine_update_scale=0
```

那么 best metric 不应该是：

```text
map_corr_wls_trans_err_mm
```

因为 WLS 当前只是 diagnostic。你的 FM-GN/WLS 总结已经显示，WLS/FM-GN 还存在窄 basin 和 rotation drift。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-fm-gn-%E5%AE%9E%E9%AA%8C%E6%80%BB%E7%BB%93.md))

best metric 应该是：

```text
full_lattice_top8_fine_rerank_mean_mm
```

或者：

```text
fine_select_final_mean_mm
```

---

# 8. 现在最应该做的实验顺序

我建议你按下面顺序推进，不要并行乱试。

## 实验 1：冻结 map，训 refine-aware coarse scorer

目标：

$$
\text{coarse top8 recall up}
$$

而不是 top1 up。

配置重点：

```text
map frozen
candidate target = multi-positive / refine-aware
topK = 8
fine-score rerank = on
fine_update_scale = 0
```

输出：

```text
top1 / top4 / top8
fine-rerank final mean
gain+
```

---

## 实验 2：GT-align map，不开 candidate-gradient

目标：

$$
\text{提升 query-map alignment，不破坏 basin}
$$

配置重点：

```text
candidate_render_score_train_map = false
candidate_score_fusion_weight = 0
only train small map decoder or adapter
```

这应该作为当前 map-side 主贡献。

---

## 实验 3：candidate adapter-only probe

目标：

$$
\text{验证 candidate-gradient 是否有小增益}
$$

不是主线。

限制：

```text
core map frozen
scorer frozen
query detached
only candidate adapter trainable
rank weight <= 0.02
strong drift / teacher / GT-align
```

只要 0.5m bucket 变坏，就停。

---

## 实验 4：随机连续 noise 验证 fixed-lattice artifact

因为 exact 0mm median 是 fixed-noise artifact，你需要额外做一个 jittered lattice eval：

```text
T0 sampled continuous noise
candidate lattice has no exact inverse candidate
```

这样才能证明 coarse topK + fine rerank 不是靠 lattice 里偶然存在 exact GT candidate。

---

# 9. 最终建议：主线这样定

当前最合理的系统定义是：

$$
\boxed{
\text{Coarse branch proposes a reliable topK basin; fine branch reranks by local alignment.}
}
$$

不是：

$$
\text{coarse top1 solves localization}
$$

也不是：

$$
\text{map-side candidate-gradient makes the map globally discriminative}
$$

所以我建议你现在把主线锁成：

```text
Phase4/Phase5-safe:
  frozen or GT-align map
  coarse topK proposal
  fine-score rerank
  no WLS update as main
  no core-map candidate-gradient
```

candidate-gradient 只保留成一组 ablation：

```text
adapter-only candidate-gradient
full-lattice target
selected-subset grad
strong drift anchor
early stop on 0.5m basin
```

一句话总结：

$$
\boxed{
\text{你现在的问题不是 topK 不够，而是把 topK selector 的 ranking gradient 直接回传给 core map field，导致 map 从“连续可 refine 特征场”变成“离散候选分类特征场”。}
}
$$

把 coarse 改成 recall generator，把 fine-score 改成官方 reranker，把 map-side candidate-gradient 限制在小 adapter 上，中 basin 被拉坏的问题才会真正缓解。

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md)

## Prompt:
2026/5/10 12:31:20

map feature field的架构是否可以改进一下？毕竟一开始设计不是为了目前的任务需求

## Response:
2026/5/10 12:31:23

我会把问题收敛到 map feature field 架构本身，判断现有 DCFF 是否适配 CPR/topK-rerank，并重点分析自由度、梯度入口与 coarse/fine 职责错位。

## Response:
2026/5/10 12:31:39

当前 DCFF 已有 latent split、spatial decoder、hash coarse、carrier/FSM 等补丁，但仍围绕 RADIO 重建堆叠。应改成稳定 base field 加独立 localization adapters/readouts。

## Response:
2026/5/10 12:36:31

可以，而且**应该改**。但我不建议把 map feature field 整体推倒重写；你现在最需要的是把它从一个“RADIO teacher reconstruction field”改成一个**任务分解的 CPR feature field**。

当前问题不是容量不够，而是架构职责错位：

$$
\text{teacher reconstruction feature}
\neq
\text{candidate topK basin feature}
\neq
\text{fine rerank / local alignment feature}
\neq
\text{WLS / pose-gradient feature}
$$

你现在把这些目标挤在同一套 map field / latent / decoder / hash MLP 里，所以 candidate-gradient 一旦进来，就会把原来可用于中 basin 的连续 feature landscape 拉坏。

---

# 1. 当前 map field 架构的问题在哪里

你现在的 DCFF 设计大致是：

$$
\text{2DGS geometry}
+
\text{per-Gaussian latent}
+
\text{deferred screen-space decoder}
$$

具体来说，当前 `DeferredCascadedRenderer` 的注释里已经写明：先 rasterize 2DGS，再 rasterize latent map，fine feature 由 latent map 通过 1×1/conv decoder 得到，coarse feature 由 position map 经过 hash grid + MLP 得到。代码注释甚至明确说 fine 是 explicit sharp boundaries，coarse 是 implicit smooth branch。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_field/dcff/deferred_renderer.py))

这个设计一开始适合做：

$$
\text{RADIO dense feature reconstruction}
$$

但现在你的任务已经变成：

$$
T_0
\rightarrow
\text{local pose lattice}
\rightarrow
\text{topK proposal}
\rightarrow
\text{fine-score rerank}
\rightarrow
T^*
$$

最新阶段总结也明确说，目前最强 CPR 路径是“local pose lattice around $T_0$ → frozen coarse candidate scorer topK → fine local-correlation rerank over topK → no forced WLS pose update”，并且 `fine_update_scale=0` 是当前主线设置。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

所以架构目标已经变了。现在 map feature field 不应该只问：

> 能不能重建 RADIO teacher feature？

而应该问：

> 能不能形成一个对候选选择、topK recall、fine rerank 和后续 pose refinement 都稳定的 pose energy landscape？

这正是当前架构没完全服务好的地方。

---

# 2. 最大结构性问题：candidate-gradient 直接改了 core map field

当前 Phase5 map finetune 配置里，candidate-gradient 不是只更新一个小 adapter，而是会打开很多 map-side 自由度，包括 `train_fine_decoder: true`、`train_coarse_fusion: true`、`train_fsm: true`、`train_hash_mlp: true`、`train_latent: true`，同时 `candidate_render_score_train_map: true`，并且 `candidate_score_fusion_weight: 0.5`。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_extract/configs/cpr_main_phase5_map_finetune_lattice.yaml))

这太大了。它等价于让一个离散候选排序 loss 同时改：

$$
z_i,\quad D_f,\quad H_{\text{hash}},\quad D_c,\quad FSM
$$

也就是让 candidate ranking loss 改整个 feature field。

你最新结果已经证明这条路危险：Phase5 GT-align 是目前唯一有用的 map-side fine-tune；candidate-gradient 的 oracle-subset 版本虽然在 0.25m/5° 上变好，但会把 0.5m/10° 的 mean 从 167.3mm 拉坏到 220.6mm，并降低 top8 recall。文档里也明确总结：candidate-gradient map fine-tune 还不能进入 mainline，因为它会 overfit compact candidate set 并 damage full-lattice ranking。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

所以我对架构的第一条建议是：

$$
\boxed{
\text{不要再让 candidate-gradient 更新 core map feature field。}
}
$$

它最多只能更新一个小的 **candidate adapter**。

---

# 3. 当前 coarse 分支本身也不够适合 CPR

你现在的 coarse 设计在 `implicit_scale` 模式下主要是：

$$
F^c_{\text{map}}
=
\text{MLP}
\left(
\text{HashGrid}(x),
\text{ScalePE}(s)
\right)
$$

也就是 position + scale 驱动。`hash_grid.py` 里也写明了两种模式：legacy 模式使用 position/hash、latent、view direction；而 `implicit_scale` 模式是 position/hash + 2D Gaussian scale positional encoding，意图上是“purely implicit, low-frequency, independent from explicit latent/view-dependent signals”。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_field/dcff/hash_grid.py))

这对 smooth semantic reconstruction 可以，但对你现在的 **local pose hypothesis scoring** 不一定够。原因是：

1. 它可能过于平滑，邻近候选 score 差异小，所以 coarse top1 不稳。
2. 它和显式 Gaussian latent 的联系弱，专家简报里也指出 coarse 路径更多被 implicit hash 分支主导，显式 latent carrier 对 coarse 分支贡献较弱。`EXPERT_REVIEW_BRIEF_2026-05-07.md`
3. candidate-gradient 一旦训练 hash MLP，很容易把世界坐标 hash 改成某种候选分类器，而不是保持连续、可 refine 的 feature field。
4. fine rerank 依赖的是局部对齐质量；coarse hash feature 和 fine feature 的 metric geometry 可能并不一致。

所以 coarse/fine 不能只靠“hash coarse + latent fine”这种原始设计自然分工。你需要显式定义 task head。

---

# 4. 我建议的新架构：Base Field + Task Adapters

最推荐的改法是：

$$
\boxed{
\text{Base RADIO field}
+
\text{Basin adapter}
+
\text{Rerank adapter}
+
\text{optional Solver adapter}
}
$$

不要把所有 loss 都打到同一个 feature 上。

---

## 4.1 Base Field：稳定的 teacher-anchored map field

保留现有 DCFF 主体，作为 base field：

$$
F^c_0(T),\quad F^f_0(T)
$$

它的职责是：

1. 重建 RADIO teacher / query-map GT align；
2. 保持 multi-view consistency；
3. 提供稳定、连续、不会被 candidate-gradient 拉坏的 feature carrier。

这个 base field 可以继续用：

- 2DGS geometry；
- per-Gaussian latent；
- fine decoder；
- coarse hash / carrier fusion；
- depth/alpha postprocess。

但它不应该直接吃大权重 candidate ranking loss。

base field 的训练目标应该是：

$$
L_{\text{base}}
=
L_{\text{RADIO}}
+
L_{\text{GT-align}}
+
L_{\text{multi-view}}
+
L_{\text{drift}}
$$

其中 $L_{\text{drift}}$ 用来防止 map feature 偏离稳定 checkpoint：

$$
L_{\text{drift}}
=
1 -
\cos
\left(
F_{\text{new}}(T),
F_{\text{old}}(T)
\right)
$$

你当前 Phase5 GT-align 结果已经说明：这种 safe map-side fine-tune 是有用的。GT-align map 在 0.25m/5° final mean 从 173.5mm 改到 146.5mm，在 0.5m/10° 从 175.1mm 改到 167.3mm，虽然幅度不大，但方向是对的。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-09-cpr-phase5-progress.md))

---

## 4.2 Basin Adapter：只服务 coarse topK，不改 base

新增一个小 residual adapter：

$$
F^{b}(T)
=
\operatorname{norm}
\left(
\operatorname{sg}(F^c_0(T))
+
\alpha_b A_b
\left(
\operatorname{sg}(F^c_0(T)),
D(T),
\alpha(T)
\right)
\right)
$$

这里：

- $F^b$：basin feature，用于 coarse topK / local hypothesis scoring；
- $F^c_0$：base coarse feature；
- $\operatorname{sg}(\cdot)$：stop-gradient；
- $A_b$：小 adapter；
- $\alpha_b$：小 residual scale，建议初始化为 0 或 0.01，最大限制到 0.1 或 0.2。

关键是：

$$
\boxed{
L_{\text{candidate}}
\rightarrow
A_b
\quad \text{but not} \quad
F^c_0,\ z_i,\ H_{\text{hash}},\ D_f
}
$$

这样 candidate-gradient 即使有偏，也只会污染 basin adapter，不会污染 core map field。

工程上你可以把它做成 1×1 + 3×3 depthwise conv 的轻量模块：

```python
class ResidualTaskAdapter(nn.Module):
    def __init__(self, dim, hidden=64, max_scale=0.1):
        super().__init__()
        self.max_scale = max_scale
        self.logit_scale = nn.Parameter(torch.tensor(-4.0))
        self.net = nn.Sequential(
            nn.Conv2d(dim + 2, hidden, 1),   # feature + depth + alpha
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base, depth, alpha, detach_base=True):
        x0 = base.detach() if detach_base else base
        aux = torch.cat([x0, depth, alpha], dim=1)
        scale = self.max_scale * torch.sigmoid(self.logit_scale)
        return F.normalize(x0 + scale * self.net(aux), dim=1)
```

然后 candidate scorer 用：

$$
F^b
$$

而不是直接用：

$$
F^c_0
$$

---

## 4.3 Rerank Adapter：只服务 fine-score rerank

fine-score rerank 现在是有效的，所以应该把它正式变成一个独立 head：

$$
F^{r}(T)
=
\operatorname{norm}
\left(
F^f_0(T)
+
\alpha_r A_r(F^f_0(T), D(T), \alpha(T))
\right)
$$

它的职责不是 WLS pose update，而是：

$$
\text{given topK candidates, select the candidate with best local alignment}
$$

训练信号：

$$
L_{\text{rerank}}
$$

$$
L_{\text{GT-align}}
$$

$$
L_{\text{local-corr}}
$$

$$
L_{\text{drift}}
$$

fine rerank 可以允许轻微回传到 $A_r$，但我仍建议先不要回传到 base fine decoder。也就是说：

```text
candidate score loss -> basin adapter only
fine-score rerank loss -> rerank adapter only
GT-align / teacher loss -> base + adapters
```

这样不同目标不会互相破坏。

---

## 4.4 Solver Adapter：暂时可选

你现在 WLS/confidence 还不稳定，FM-GN / WLS 更像 diagnostic，不是主线。那就不要让 solver loss 大规模改变 map field。

如果你要保留 solver 分支，可以单独加：

$$
F^{s}(T)
$$

和一个 confidence head：

$$
C^s(T)
$$

它只服务：

- WLS confidence；
- pose Jacobian observability；
- local feature-metric alignment。

但目前我建议先不要把它作为主线。主线还是：

$$
\text{coarse topK}
+
\text{fine-score rerank}
+
\text{no forced WLS update}
$$

---

# 5. 更合理的 map field v2 输出

当前 `render_feature_bundle_at_pose` 返回类似：

```python
{
    "fine_features": ...,
    "coarse_features": ...,
    "depth": ...,
    "fsm_spatial_conf": ...,
    "fsm_channel_weights": ...
}
```

我建议 v2 改成：

```python
{
    "base_coarse": F_c0,
    "base_fine": F_f0,

    "basin_features": F_b,
    "rerank_features": F_r,
    "solver_features": F_s,          # optional

    "depth": depth,
    "alpha": alpha,

    "basin_conf": C_b,
    "rerank_conf": C_r,
    "solver_conf": C_s,              # optional

    "feature_drift": optional_debug,
}
```

这样训练和评估时可以明确指定：

```yaml
candidate_score_fusion_feature: basin_features
fine_select_feature: rerank_features
wls_feature: solver_features
teacher_anchor_feature: base_fine/base_coarse
```

不要再用一个 `"coarse"` 或 `"fine"` 承担所有任务。

---

# 6. latent 也应该改成 task-split latent

现在 `HybridGaussianModel` 是单个 per-Gaussian latent：

$$
z_i \in \mathbb{R}^{d}
$$

代码里也是标准 2DGS geometry 加一个 learnable latent embedding，并且 latent 会参与保存、加载和 densification。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_field/dcff/hybrid_gaussian.py))

虽然 `DeferredCascadedRenderer` 已经支持 `fine_latent_dim` 和 `coarse_latent_dim` 的 split latent，但是从任务角度，我建议你显式改成：

$$
z_i =
[z_i^{base}, z_i^{basin}, z_i^{precision}]
$$

或者至少：

$$
z_i =
[z_i^{coarse}, z_i^{fine}]
$$

职责如下：

| latent | 用途 | 是否允许 candidate-gradient |
|---|---|---|
| $z^{base}$ | teacher reconstruction / stable carrier | 否 |
| $z^{basin}$ | topK candidate proposal adapter | 只在 adapter-only 路径中可训练 |
| $z^{precision}$ | local rerank / correspondence | 不吃 candidate rank，只吃 GT-align/local-corr |

但我不建议第一步就真的让 $z^{basin}$ 直接可训练。更稳的是：

$$
z^{basin}
\rightarrow
A_b
$$

先训练 adapter，不动 latent。等 adapter 证明不伤 0.5m/10° bucket，再考虑打开 very-low-LR latent。

---

# 7. coarse branch 不建议继续用 pure implicit hash 作为唯一主干

当前 `implicit_scale` coarse branch 的优点是平滑、低频、速度可控。问题是它对 CPR topK selector 来说可能太“场景坐标函数化”，容易出现：

$$
\text{candidate rank loss}
\rightarrow
\text{hash MLP 变成局部候选分类器}
$$

我建议 coarse branch 改成 **hybrid basin feature**：

$$
F^b
=
\operatorname{norm}
\left(
F_{\text{hash}}(x,s)
+
\lambda_z F_{\text{carrier}}(z^c)
+
\lambda_g F_{\text{geom}}(d,\alpha,n)
+
A_b(\cdot)
\right)
$$

其中：

- $F_{\text{hash}}$：保留原来的 implicit smooth field；
- $F_{\text{carrier}}$：来自 rasterized Gaussian latent，提供可见结构和局部 anchor；
- $F_{\text{geom}}$：depth / alpha / normal / scale 的轻量编码；
- $A_b$：candidate-only residual adapter。

你代码里其实已经有一些可以复用的部件，比如 `CarrierResidualCoarseFusion` 和 `SpatialDirectCoarseDecoder`。`CarrierResidualCoarseFusion` 的注释就是“在 implicit coarse field 上加 carrier-conditioned residual”，`SpatialDirectCoarseDecoder` 则是从 rasterized coarse latent 直接解码 coarse feature。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_field/dcff/deferred_renderer.py))

所以不是从零开始写。你可以先试：

```yaml
coarse_decoder:
  mode: carrier_residual
```

或者：

```yaml
coarse_decoder:
  mode: spatial_direct
```

但要注意：**carrier_residual / spatial_direct 也不能直接吃大权重 candidate-gradient**。它们应该先用于 GT-align / teacher / topK recall 的 safe training，candidate-gradient 只走 basin adapter。

---

# 8. FSM 也应该降级，不要让它承担主分工

当前 Feature Selection Module 的设计是：spatial confidence gate、channel selection、cross-attention fusion，在 renderer 之后融合 fine/coarse features。代码注释里也明确了这三部分。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_field/dcff/feature_selection.py))

这个模块有价值，但现在不应该让 FSM 变成“所有任务的自动路由器”。原因是：

1. candidate score、fine rerank、WLS confidence 三个目标冲突；
2. Gumbel/channel routing 如果没有明确任务监督，容易学成 shortcut；
3. candidate-gradient 打到 FSM，也可能改变 fine rerank 依赖的 feature geometry。

我建议 FSM v2 拆成两个简单 gate：

$$
C_b(u)
$$

$$
C_r(u)
$$

分别服务：

- basin candidate scoring；
- fine-score rerank。

不要用同一个 FSM 同时改 coarse 和 fine。也不要让 candidate loss 更新 fine FSM。

---

# 9. 最小可行重构方案

不要一次性大改。我建议按三步做。

---

## Step A：保留当前 DCFF，新增 basin/rerank adapter

先不动 2DGS、latent、hash grid、fine decoder。只新增两个 residual head：

```text
base_coarse -> basin_adapter -> basin_features
base_fine   -> rerank_adapter -> rerank_features
```

训练规则：

```text
candidate rank loss -> basin_adapter only
fine-score rerank loss -> rerank_adapter only
GT-align / teacher loss -> base + adapters, or base only
```

配置上：

```yaml
map_field_v2:
  enabled: true

  basin_adapter:
    enabled: true
    detach_base: true
    max_residual_scale: 0.10
    input_feature: base_coarse
    use_depth_alpha: true

  rerank_adapter:
    enabled: true
    detach_base_for_candidate: true
    max_residual_scale: 0.10
    input_feature: base_fine
    use_depth_alpha: true

map_supervision:
  train_hash_mlp: false
  train_latent: false
  train_fine_decoder: false
  train_coarse_fusion: false
  train_fsm: false

  train_basin_adapter: true
  train_rerank_adapter: true
```

这是最关键的第一版。

---

## Step B：用 GT-align 训练 base field，candidate loss 不碰 base

safe map-side fine-tune 可以继续，但目标要改成：

```yaml
candidate_render_score_train_map: false
candidate_score_fusion_weight: 0.0

query_fine_weight: 0.10
query_coarse_weight: 0.05
rendered_teacher_fine_weight: 0.001
rendered_teacher_coarse_weight: 0.001

train_fine_decoder: true
train_coarse_fusion: true
train_hash_mlp: false
train_latent: false
```

这一步训练的是：

$$
F^c_0,\ F^f_0
$$

让 base field 更贴近 query-map CPR feature space。

---

## Step C：candidate-gradient 只训练 basin adapter

candidate loss 权重降到很小：

```yaml
candidate_score_fusion_weight: 0.02
candidate_score_fusion_pairwise_rank_weight: 0.02
```

并且：

```yaml
train_basin_adapter: true

train_hash_mlp: false
train_latent: false
train_fine_decoder: false
train_coarse_fusion: false
train_fsm: false
```

再加 drift：

```yaml
map_feature_drift_weight: 0.10
```

这时 candidate-gradient 的作用只是轻微塑形：

$$
F^b
$$

而不是改：

$$
F^c_0,\ F^f_0
$$

---

# 10. 你应该加的硬性 gradient check

这个很重要。否则你以为 candidate-gradient 只训练 adapter，实际可能还是打进 base field。

加测试：

```python
loss_candidate.backward()

assert basin_adapter.some_weight.grad is not None

assert dcff_renderer.hash_grid.mlp[0].weight.grad is None
assert dcff_renderer.fine_decoder.decoder[0].weight.grad is None
assert gaussians._latent.grad is None
assert dcff_renderer.coarse_carrier_fusion is None or \
       all(p.grad is None for p in dcff_renderer.coarse_carrier_fusion.parameters())
```

如果你用 `detach_base=True`，那么 candidate loss 的梯度链路应该是：

$$
L_{\text{candidate}}
\rightarrow
S_{\text{candidate}}
\rightarrow
F^b
\rightarrow
A_b
$$

而不是：

$$
L_{\text{candidate}}
\rightarrow
z_i,\ H_{\text{hash}},\ D_f
$$

---

# 11. 训练目标也要跟架构匹配

新架构下 loss 应该写成：

$$
L =
L_{\text{base}}
+
L_{\text{basin}}
+
L_{\text{rerank}}
+
L_{\text{regularize}}
$$

其中：

## Base loss

$$
L_{\text{base}}
=
\lambda_t L_{\text{RADIO}}
+
\lambda_g L_{\text{GT-align}}
+
\lambda_{mv}L_{\text{multi-view}}
$$

## Basin loss

不要用强 top1 CE。用 multi-positive topK loss：

$$
P(q) =
\{k \mid d(T_k,T_{gt}) < \delta\}
$$

$$
L_{\text{basin}}
=
-\log
\frac{
\sum_{k\in P(q)} \exp(s_k/\tau)
}{
\sum_j \exp(s_j/\tau)
}
$$

目标是：

$$
\text{topK recall}
$$

不是 top1。

## Rerank loss

fine-score rerank 应该学 recoverability：

$$
k^+
=
\arg\min_k
d(T_k^{\text{selected/fine}}, T_{gt})
$$

而不是单纯 pose-nearest。

也就是训练它选：

> 哪个 candidate 最容易被当前 fine branch 对齐。

## Regularization

$$
L_{\text{drift}}
=
1-\cos(F_{\text{adapter}},F_{\text{base}})
$$

$$
L_{\text{scale}}
=
\|\alpha_b\|^2+\|\alpha_r\|^2
$$

$$
L_{\text{smooth-basin}}
$$

只用于 basin feature，不要强行平滑 precision feature。

---

# 12. 推荐的 v2 结构图

可以把 map field v2 理解成这样：

```text
                 2DGS geometry
                      |
       rasterize latent / depth / alpha / scale / pos
                      |
              DCFF base renderer
             /                  \
      base_coarse             base_fine
          |                       |
   Basin Adapter            Rerank Adapter
          |                       |
   basin_features          rerank_features
          |                       |
 local pose topK          fine-score rerank
          \                       /
             selected refined pose
```

对应梯度：

```text
teacher / GT-align loss
        -> base renderer, optional adapters

candidate topK loss
        -> basin adapter only

fine rerank loss
        -> rerank adapter only

WLS / solver loss
        -> optional solver adapter only
```

这是和当前任务需求匹配的结构。

---

# 13. 更激进但有价值的长期改法：landmark / cell-level basin field

如果你后面想把 CPR 和检索合并，我建议 coarse/basin 分支再进一步改成 **landmark token field**。

现在 coarse feature 仍是 dense pixel feature：

$$
F^b(T) \in \mathbb{R}^{H\times W\times C}
$$

但 topK candidate scoring 本质上不需要每个像素都精确。它需要稳定的 layout / landmark evidence。可以把 map 聚成若干 3D cells / Gaussian clusters：

$$
\ell_m =
\{x_m,\ f_m,\ r_m\}
$$

渲染时输出 visible landmark tokens 或 low-res token map：

$$
\mathcal{L}(T)
=
\{\ell_m \mid \ell_m \text{ visible under } T\}
$$

candidate scoring 用 token matching：

$$
s_k
=
S(Q^b,\mathcal{L}(T_k))
$$

优点：

1. topK scoring 更稳定；
2. 不容易被局部纹理噪声影响；
3. 和未来 retrieval pose bank 更自然；
4. candidate-gradient 不需要改 dense fine feature。

但这个改动比较大。我建议先做 adapter v2，等结果稳定后再考虑 landmark basin field。

---

# 14. 应该怎么验证架构改进有效

你不要只看 cosine，也不要只看 compact subset。必须看 full-lattice CPR 指标。

建议 ablation：

| 版本 | 目的 |
|---|---|
| current frozen map | baseline |
| Phase5 GT-align map | safe map-side improvement |
| full candidate-gradient map | negative control |
| base + basin adapter only | 验证 candidate-gradient 是否不再伤中 basin |
| base + rerank adapter only | 验证 fine-score rerank 是否增强 |
| base + basin adapter + rerank adapter | v2 主结果 |
| carrier_residual coarse + adapters | 验证 coarse carrier 是否改善 topK |
| spatial_direct coarse + adapters | 验证显式 coarse latent 是否比 pure hash 更稳 |

指标：

```text
0.25m/5° final mean
0.5m/10° final mean
gain+
top4 recall
top8 recall
fine-rerank final mean
feature drift from base
candidate score margin
```

尤其要看：

$$
0.25m \text{ 变好时，} 0.5m \text{ 不能变坏}
$$

这就是当前 candidate-gradient 失败的关键。

---

# 15. 最直接的建议

我会这样改：

## 第一阶段：不改 core，只加 adapter

```text
DCFF base field frozen or GT-align fine-tuned
+
basin adapter
+
rerank adapter
```

candidate loss 只训练 basin adapter。

fine-score loss 只训练 rerank adapter。

这一步最应该马上做。

---

## 第二阶段：把 coarse 从 pure implicit hash 改成 hybrid coarse

优先试：

```yaml
coarse_decoder:
  mode: carrier_residual
```

其次试：

```yaml
coarse_decoder:
  mode: spatial_direct
```

但 candidate-gradient 仍然只进 adapter，不进 core。

---

## 第三阶段：再考虑 landmark/token basin field

这是未来接可微检索和完整 localization 的方向，不是当前最小改动。

---

# 16. 一句话结论

是的，map feature field 架构需要改。当前 DCFF 更像“RADIO reconstruction field”，而你现在需要的是：

$$
\boxed{
\text{teacher-anchored base field}
+
\text{candidate topK basin adapter}
+
\text{fine-score rerank adapter}
+
\text{可选 solver confidence head}
}
$$

不要再让 candidate-gradient 直接训练 hash MLP、latent、fine decoder、coarse fusion 和 FSM。它应该只进入一个小的 residual basin adapter。这样才能保住中 basin 的连续性，同时让 coarse topK 和 fine rerank 各自变强。

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_field/dcff/deferred_renderer.py)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)