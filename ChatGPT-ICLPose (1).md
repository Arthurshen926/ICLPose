# ICLPose

**User:** Anonymous  
**Created:** 5/16/2026 12:54:37  
**Updated:** 5/17/2026 0:19:34  
**Exported:** 5/17/2026 0:38:36  
**Link:** [https://chatgpt.com/c/6a07f88d-362c-83ea-a904-b9f42272c2e9](https://chatgpt.com/c/6a07f88d-362c-83ea-a904-b9f42272c2e9)  

## Prompt:
2026/5/16 23:54:14

按照我们讨论的内容又推进了一轮，git仓库已更新。但感觉还没达到预期目标。记住：最终本工作的目标是要投稿顶刊的，所以不光是逻辑合理，方法有创新和精度有提升这三条条件都需要满足。下一步应该如何进一步修改并优化？最后要给一个预期目标或者终止条件，我要让codex的goal模式帮我推进

## Response:
2026/5/16 23:54:16

我会按投稿级证据链复核最新主线，只看 POFD/q50/q75 的新记录、关键脚本/config 与 score surface 变化。重点给出 Codex goal 可执行的目标、验收指标和终止条件。

## Response:
2026/5/16 23:54:19

我看到 Stage3a 已将 q50 从约 `0.362m` 推到 `0.223–0.225m`，但 radius/scale 与 raw flow-NCE 已接近平台。下一步应直接监督 pair-matcher logits/offset 与候选 pose-energy 排序，专门压制剩余 hard wrong candidates。

## Response:
2026/5/17 00:02:06

情况确实还没到“可以写顶刊主结果”的程度，但我不认为现在是坏局面。**Stage3a 已经把一个关键瓶颈实锤了：q50 的主要问题不是特征完全没学到，而是 high-res small-radius score path 看不到中等扰动下的正确证据。** 你最新记录里，val128 从 Stage2d 的 `0.362m / top1 0.379 / spearman 0.113` 提升到 Stage3a 的约 `0.225m / top1 0.711 / spearman 0.535`，这已经是非常实质的进展；但 r16 之后基本边际收益，raw flow-NCE 也没有超过 Stage3a，说明继续调 radius/scale 或 raw correlation loss 的收益已经接近平台。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-16-pofd-stage3a-radius-scale.md))

我的判断是：

> **下一步不要大改系统，也不要回到 selector/confidence/Fisher/map-side adaptation。应该把方法核心升级为：Pair-Matcher-Aware Pose-Energy Distillation。**

也就是：不要再让 POFD 只学“feature 看起来好不好”或“raw correlation 有没有峰值”，而是直接监督最终用于候选排序的 **pair-matcher offset logits / pose score surface**。

---

## 1. 当前阶段的真实状态

现在 POFD 主线的状态可以概括成三句话：

**第一，逻辑链条已经比之前清楚。**  
你已经从“从 RADIO 蒸馏一个定位特征”收敛成了“给定初值，学习能扩大 pose refinement basin 的 localization feature”。这比完整 retrieval + refinement 的大而全 claim 更稳。RADIO / AM-RADIO 本身就是 multi-teacher VFM distillation，把 CLIP、DINOv2、SAM 等能力压到统一 backbone 中，因此你的科学问题可以定位为：这些 dense VFM feature 中哪些信息对位姿能量面有用。([GitHub](https://github.com/nvlabs/radio))

**第二，精度有明显提升，但还没到强主结果。**  
`0.362m -> 0.225m` 是强进展，但对于 q50 refinement 论文主结果，`0.225m` 仍然比较尴尬：它比 `<0.20m` 目标差一截，而且和 oracle gap 仍有约 `9cm`。从顶刊角度，最好把 full-val q50 推到 `0.18m~0.20m` 区间，同时证明 q10/q25 不退化。

**第三，创新性目前还不够硬。**  
Stage3a 的“低分辨率 + 大半径 score path”是重要诊断和有效工程改造，但单独作为顶刊核心贡献偏弱。已有 GSFF 这类工作已经做了 Gaussian feature field + 2D encoder + feature/segmentation alignment + pose refinement，GS-CPR 也已经用 3DGS render + dense correspondence 做相机位姿 refinement。你的论文必须明确胜出点不是“也做 3DGS refinement”，而是：**从视觉基础模型特征中蒸馏出对 pose score surface 排序正确的特征，并用 hard pose-energy supervision 改善中等 basin。** ([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Pietrantoni_Gaussian_Splatting_Feature_Fields_for_Privacy-Preserving_Visual_Localization_CVPR_2025_paper.pdf))

所以下一步的目标不是再找一个小 trick，而是把方法变成一个能写成论文核心模块的东西。

---

## 2. 下一步主方法：Pair-Matcher-Aware Pose-Energy Distillation

你现在最该新增的不是新的 selector，而是一个直接作用在 pair matcher score surface 上的 loss family。我建议命名为：

```text
PMED: Pair-Matcher-Aware Pose-Energy Distillation
```

或者作为 POFD 的 Stage3 核心：

```text
POFD + PMED
```

它解决的问题是：

```text
raw feature / raw correlation 好看 ≠ pair-matcher 最终候选排序正确
```

你的 Stage3c raw flow-NCE 失败已经说明这一点：它优化的是 raw local correlation，但最终排序路径是 `PairConditionedLocalMatcher` 的 score surface，所以 loss 和决策层不一致。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-16-pofd-stage3a-radius-scale.md))

### 2.1 具体要监督什么？

对每个 query，有 top-K candidate renders。当前系统会给每个 candidate 一个 score，然后选最高分。现在剩余问题是：

```text
某些 hard wrong candidate 的 score 仍然高于 oracle/good candidate。
```

因此 loss 应该直接约束：

```text
score(query, good candidate) > score(query, hard wrong candidate)
```

但不要只做 candidate-level hinge。更关键的是 pair matcher 内部的 offset logits：

```text
L_c(p, Δu)
```

其中：

- `c` 是 candidate；
- `p` 是 score-grid 上的像素；
- `Δu` 是 pair matcher 搜索窗口内的 offset；
- `L_c(p, Δu)` 是 pair matcher 判断 query-map 对应的局部匹配 logit。

你要让 good candidate 在几何正确 offset 上有高 logit，让 hard wrong candidate 的最高 logit 被压低。

---

## 3. 推荐新增两个核心 loss

### Loss A：Pair-Matcher GeoNCE

对每个 query，选：

```text
c_pos = topK 中 GT pose cost 最小的 candidate
c_neg = pose cost 明显更差但当前 score 最高的 candidate
```

然后对 `c_pos` 的可见像素采样。对每个像素 `p`，用 GT pose / world-position map / depth 投影得到正确 offset：

```text
Δu* = projection_under_GT(world_position_from_candidate_render(p)) - p
```

只保留满足这些条件的像素：

```text
1. 投影落在图像内；
2. offset 落在 pair matcher radius 内；
3. candidate render alpha / valid mask 有效；
4. depth consistency 通过；
5. 非遮挡或遮挡不严重。
```

然后做 offset-level cross entropy / InfoNCE：

```text
L_geo_offset =
    CE( logits_pos_candidate[p, :] , target_offset = Δu* )
```

再加 hard wrong 抑制：

```text
L_geo_hard =
    softplus( max_Δ logits_neg_candidate[p, Δ]
              - logits_pos_candidate[p, Δu*]
              + margin )
```

最终：

```text
L_pair_geo_nce = L_geo_offset + λ_hard L_geo_hard
```

这一步是关键。它不是 raw flow-NCE，而是直接监督 pair matcher 的 offset logits。

---

### Loss B：Pose-Cost-Shaped Listwise Ranking

现在 candidate ranking 不应该只把 `argmin cost` 当唯一正样本。q50 下 candidate 的 pose cost 有连续结构，应该用 soft target distribution：

```text
q_i = softmax( - cost_i / τ_cost )
p_i = softmax( score_i / τ_score )
L_cost_listwise = KL(q || p)
```

再叠加 online hard-negative margin：

```text
L_hard_rank =
    softplus( score_hard_wrong - score_good + margin )
```

这样可以避免模型只学“选一个 best”，而是让整个 score surface 的排序更符合真实 pose cost。你现在的 spearman 已经到 `0.53~0.55`，说明排序相关性可被进一步优化；目标应该是把 spearman 推到 `0.60+`。

---

## 4. 为什么这比继续调 radius/flow 更值得做？

因为 Stage3a 已经回答了：

```text
score path 的搜索范围问题基本解决了。
```

最新 full val128 上：

```text
Stage2d baseline: 0.362m
Stage3a r12/r16: 约 0.223m~0.225m
```

但 r16 继续提升已经只有 1~2mm 级别，17x30 低分辨率反而退化；raw flow-NCE 也只是边际 1mm 左右，没有超过 Stage3a。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-16-pofd-stage3a-radius-scale.md))

所以剩余问题不是“窗口再大一点”，而是：

```text
pair matcher 看到了正确候选，但还没有稳定学会把 good candidate 的 offset evidence 转成更高 pose score。
```

这正是 PMED 要解决的问题。

---

## 5. 具体代码修改建议

### 5.1 新增配置项

在 `train_nvs_pose_feature_adapter.py` 或对应 config schema 中新增：

```yaml
pair_matcher_geo_nce_weight: 0.05
pair_matcher_geo_hard_weight: 0.05
pair_matcher_geo_candidate_mode: best_and_score_hard_negative
pair_matcher_geo_num_pixels: 512
pair_matcher_geo_cost_gap_m: 0.12
pair_matcher_geo_margin: 0.10
pair_matcher_geo_temperature: 0.07

cost_shaped_rank_weight: 0.5
cost_shaped_rank_temperature_m: 0.10
score_temperature: 0.07

geo_visibility_alpha_min: 0.2
geo_depth_consistency_thresh_m: 0.10
geo_require_offset_in_radius: true
geo_min_valid_pixels: 64
```

默认先保守，不要一上来把 loss 权重拉大。

---

### 5.2 新增 pair-matcher logits hook

现在 Stage3c 的问题是 loss 作用在 raw local correlation。下一步要从 pair matcher 中拿到：

```python
offset_logits: [B, K, Hs, Ws, O]
candidate_scores: [B, K]
```

其中 `O = (2r + 1)^2`。

如果当前 pair matcher 只返回 candidate score，需要改成可选返回：

```python
return_debug=True
```

输出：

```python
{
    "candidate_scores": scores,
    "offset_logits": offset_logits,
    "offset_grid": offset_grid,
    "valid_mask": valid_mask,
}
```

注意：这个 debug 输出应该只在训练/eval loss 需要时启用，不要污染默认路径。

---

### 5.3 实现 target offset 计算

对 candidate render 中每个 score-grid 像素 `p`，拿到对应 world point：

```python
X = candidate_world_position[p]
```

用 GT query pose 投影到 query image：

```python
u_gt = project(K, T_gt, X)
```

再换算到 score grid：

```python
u_gt_s = u_gt / score_stride
p_s = p / score_stride
offset = round(u_gt_s - p_s)
```

然后检查：

```python
abs(offset_x) <= radius
abs(offset_y) <= radius
```

这一步必须严格处理 intrinsics scaling。你记录里已经提到 Stage3c 修过低分辨率 score grid 下的 intrinsics 和 world-position map resize，下一步要继续沿用这个逻辑。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-16-pofd-stage3a-radius-scale.md))

---

### 5.4 新增诊断指标

不要只看最终 pred cost。新增：

```text
geo_offset_valid_frac
geo_offset_ce
geo_offset_top1_acc
pos_logit_minus_max_wrong_logit
pos_score_minus_hardneg_score
hardneg_pose_cost_m
hardneg_selected_frac
cost_rank_kl
per_query_failure_dump
```

特别重要的是：

```text
pos_logit_minus_max_wrong_logit
```

如果这个指标涨了，但 pred cost 不涨，说明 pair matcher 内部学到了对应，但 candidate aggregation 有问题。  
如果这个指标不涨，说明 offset-level supervision 没打进去。

---

## 6. 推荐实验矩阵

下一轮不要铺太多 config。只跑 6 个：

| 编号 | 实验 | 目的 | Promote 条件 |
|---|---|---|---|
| S0 | Stage3a 当前 best 复评 | 固化 baseline | q50 val128 约 `0.223~0.225m` |
| D1 | cost-shaped listwise only | 看 candidate score calibration 是否有效 | 相比 S0 至少 `-8mm` |
| D2 | pair-matcher GeoNCE only | 看 offset logits 监督是否有效 | `geo_offset_top1` 明显上升，pred 至少 `-8mm` |
| D3 | GeoNCE + cost-shaped listwise | 主候选 | q50 `≤0.205m` 或接近 |
| D4 | D3 checkpoint + r16 scorer eval-only | 验证 r16 是否仍可吃到收益 | q50 `≤0.200m` |
| D5 | q10/q25 regression eval | 防止中 basin 提升损害小 basin | q10/q25 不退化 |

训练建议：

```text
start checkpoint:
result/result/feature_extract/pofd_stage3a_val128_r12_continue_b2_noamp_s40_20260516/checkpoints/best.pth

primary score path:
34x60 / r12 train

eval score path:
34x60 / r12
34x60 / r16 eval-only

batch:
先 batch=1/2 no-amp，保证稳定，不追求吞吐
```

---

## 7. 暂时不要做的事

这些方向我建议继续冻结：

```text
1. 不要回到 trainable topK selector。
2. 不要继续调 confidence scalar。
3. 不要把 Fisher/logdet 当主 loss。
4. 不要加强 teacher anchor。
5. 不要解冻 map-side latent / geometry。
6. 不要继续扩大 radius 或继续降分辨率。
7. 不要继续 raw flow-NCE 权重扫描。
```

原因很简单：这些已经被当前实验部分否定，或者收益已经接近平台。下一步必须打到最终决策层：pair matcher offset logits + candidate pose-energy ranking。

---

## 8. 顶刊角度的下一阶段目标

你现在不能只设一个 `<0.20m`，因为顶刊需要同时满足 **逻辑、创新、精度**。我建议设三级目标。

### 目标 A：技术推进目标

这是 Codex goal mode 当前要追的目标：

```text
q50 val128:
pred_cost_m <= 0.200
top1_acc >= 0.78
spearman >= 0.58
succ@25cm/10deg >= 0.80
oracle_gap_m <= 0.070

q10/q25:
pred_cost_m <= 0.075
succ@10cm/5deg >= 0.90
不能比当前 Stage2/Stage3 小 basin 主结果明显退化。
```

这个目标达成，说明 POFD + PMED 的主方法成立。

---

### 目标 B：论文主结果目标

技术目标达成后，必须扩大评估：

```text
full validation / uncapped validation:
q50 pred_cost_m <= 0.20

3 seeds:
均值 <= 0.205m
最好 <= 0.195m
std <= 0.015m

ablation:
Stage2d -> Stage3a -> Stage3d 每一步都有明确提升

failure analysis:
剩余失败主要来自 oracle candidate 本身不足或遮挡/极端重复结构，而不是 score surface 随机失效。
```

如果只在 val128 上好，顶刊还不够。

---

### 目标 C：投稿级目标

投稿前至少需要：

```text
1. OldHospital 主结果完整；
2. 至少一个额外 scene / dataset；
3. 和 RADIO raw、teacher reconstruction、Stage2d、Stage3a、GSFF/GS-CPR-style baselines 对比；
4. oracle/deploy/ablation eval mode 明确隔离；
5. basin curve 覆盖 q10/q25/q50；
6. 可视化 pair-matcher score surface、hard negative 抑制、offset logit sharpening；
7. 不依赖 oracle retrieval claim。
```

已有 GSFF 和 GS-CPR 都在 pose refinement / Gaussian feature field 方向占了位置，所以你的论文必须证明“VFM feature 经 pose-energy 蒸馏后得到更好的 score surface”，而不是只证明“我也能用 3DGS 做 refinement”。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Pietrantoni_Gaussian_Splatting_Feature_Fields_for_Privacy-Preserving_Visual_Localization_CVPR_2025_paper.pdf))

---

## 9. 终止条件

你需要给 Codex 一个明确的 stop rule，否则它会继续无意义扫描 config。

我建议这样设：

### Promote 条件

满足任一：

```text
1. q50 val128 pred_cost_m <= 0.200
   且 top1 >= 0.78
   且 spearman >= 0.58
   且 q10/q25 不退化；

或：

2. q50 val128 pred_cost_m <= 0.205
   且 oracle_gap_m <= 0.075
   且 succ@25cm/10deg >= 0.80
   且 D3/D4 相对 Stage3a 至少提升 20mm。
```

满足后，进入 full validation + multi-seed，不再继续改 loss。

---

### Continue 条件

```text
0.205 < q50 pred_cost_m <= 0.215
且 pair-matcher GeoNCE 内部指标明显改善：
geo_offset_top1_acc 上升
pos_logit_minus_max_wrong_logit 上升
hardneg score margin 上升
```

这种情况说明方向有效，但 loss/filter 还没调好。可以继续一轮 visibility filter、pixel sampling、temperature、hard negative gap 的小规模调整。

---

### Stop / Pivot 条件

满足任一就停止 Stage3d：

```text
1. 连续 3 个受控 config 相比 Stage3a best 提升 < 5mm；
2. 最好结果仍然 >= 0.220m；
3. geo_offset 指标改善，但 candidate pred_cost 不改善；
4. q50 改善但 q10/q25 明显退化；
5. 训练只在 val128 子集有效，full validation 崩掉。
```

如果触发 stop，不要继续调 POFD feature loss。下一步 pivot 到：

```text
selected candidate 后的 iterative pose refinement / solver update
```

也就是从“候选排序”转向“选中后如何连续优化位姿”。

---

## 10. 可直接给 Codex goal mode 的任务描述

下面这段可以直接作为 Codex 的 goal：

```text
Goal: Implement and validate POFD Stage3d: Pair-Matcher-Aware Pose-Energy Distillation for q50 basin refinement.

Context:
Current POFD mainline is single localization feature, no coarse/fine retrieval, no trainable topK selector.
Current best full val128 q50 result is Stage3a:
- checkpoint: result/result/feature_extract/pofd_stage3a_val128_r12_continue_b2_noamp_s40_20260516/checkpoints/best.pth
- q50 val128 with r16 score path: pred_cost_m ≈ 0.2227, top1 ≈ 0.7188, spearman ≈ 0.5498
Stage3a fixed the radius/scale bottleneck. Further radius/scale tuning has marginal gains. Raw flow-NCE did not improve because it optimizes raw correlation rather than the PairConditionedLocalMatcher score surface.

Hard constraints:
- Do not reintroduce trainable topK selector.
- Do not tune confidence scalar.
- Do not use Fisher/logdet as main loss.
- Do not strengthen teacher anchor.
- Do not unfreeze map-side latent/geometry.
- Do not continue pure radius/scale sweeps.
- Preserve eval_mode oracle/deploy/ablation separation.
- Remove/avoid validation caps when claiming val128/full-val metrics.
- Run regression tests after code changes.

Implementation:
1. Add pair-matcher-aware GeoNCE loss operating directly on PairConditionedLocalMatcher offset logits, not raw local correlation.
2. Expose optional debug outputs from the pair matcher:
   - candidate_scores
   - offset_logits [B, K, Hs, Ws, O]
   - offset_grid
   - valid masks
3. Compute GT target offsets for candidate render pixels:
   - use candidate world-position/depth map
   - project 3D point into GT query camera
   - scale intrinsics correctly for score_feature_hw
   - convert to score-grid offset
   - filter invalid/out-of-radius/occluded/depth-inconsistent pixels
4. Candidate selection for the loss:
   - positive = topK candidate with minimum GT pose cost
   - hard negative = pose cost worse by at least 0.12m and currently highest predicted score
5. Losses:
   - pair_matcher_geo_offset_ce
   - pair_matcher_geo_hard_margin
   - cost_shaped_listwise_KL over candidates, with target q_i = softmax(-cost_i / tau_cost)
   - existing Stage3a rank/teacher-quality losses retained unless they conflict
6. Add metrics:
   - geo_offset_valid_frac
   - geo_offset_top1_acc
   - geo_offset_ce
   - pos_logit_minus_max_wrong_logit
   - pos_score_minus_hardneg_score
   - hardneg_pose_cost_m
   - cost_rank_kl
   - per-query failure dump for selected wrong candidates
7. Add configs:
   - pofd_stage3d_cost_listwise_q50_val128.yaml
   - pofd_stage3d_pairmatcher_geonce_q50_val128.yaml
   - pofd_stage3d_pairmatcher_geonce_costlist_q50_val128.yaml
   - pofd_stage3d_q10q25_regression_eval.yaml

Experiment matrix:
S0: Re-evaluate Stage3a best on q50 val128.
D1: Cost-shaped listwise only.
D2: Pair-matcher GeoNCE only.
D3: GeoNCE + cost-shaped listwise.
D4: D3 checkpoint with r16 score path eval-only.
D5: q10/q25 regression eval.

Primary target:
Promote Stage3d only if:
- q50 val128 pred_cost_m <= 0.200
- top1_acc >= 0.78
- spearman >= 0.58
- succ@25cm/10deg >= 0.80
- oracle_gap_m <= 0.070
- q10/q25 pred_cost_m <= 0.075 and succ@10cm/5deg >= 0.90

Continue condition:
If q50 pred_cost_m is between 0.205 and 0.215 and geo_offset/score-margin metrics improve, do one additional small tuning round for visibility filtering, temperature, and hard-negative cost gap.

Stop condition:
Stop Stage3d and write a no-go report if:
- 3 controlled configs improve less than 5mm over Stage3a best, or
- best q50 pred_cost_m remains >= 0.220, or
- geo_offset metrics improve but candidate ranking does not, or
- q10/q25 regresses significantly.

Deliverables:
- Updated implementation and configs.
- Updated docs/superpowers/plans/2026-05-16-pofd-stage3d-pmed.md
- Updated basin report JSON/MD.
- Full table comparing Stage2d, Stage3a, D1, D2, D3, D4, D5.
- Test result summary.
```

---

## 11. 最终建议

你现在的主线不要推翻。**POFD 是对的，Stage3a 也是对的；但 Stage3a 只是把 score path 的尺度/搜索窗口补齐，还不是足够强的论文核心。**

下一步要把核心创新落在：

```text
Pair-Matcher-Aware Pose-Energy Distillation
```

也就是：

```text
从 VFM feature 蒸馏到 localization feature，
不是让 descriptor 更像 teacher，
也不是让 raw correlation 更尖锐，
而是让 pair matcher 的 offset logits 和 candidate score distribution
直接服从真实 pose cost surface。
```

如果 Stage3d 能把 q50 full val128 推到 `≤0.20m`，同时 q10/q25 不退化，这条线就值得继续扩到 full validation、multi-seed、cross-scene 和论文 ablation。  
如果 Stage3d 仍卡在 `0.22m` 左右，就应该停止 feature-loss 方向，转向 selected candidate 后的 iterative pose refinement。

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-16-pofd-stage3a-radius-scale.md)
- [GitHub - NVlabs/RADIO: Official repository for "AM-RADIO: Reduce All Domains Into One" · GitHub](https://github.com/nvlabs/radio)
- [Gaussian Splatting Feature Fields for (Privacy-Preserving) Visual Localization](https://openaccess.thecvf.com/content/CVPR2025/papers/Pietrantoni_Gaussian_Splatting_Feature_Fields_for_Privacy-Preserving_Visual_Localization_CVPR_2025_paper.pdf)

## Prompt:
2026/5/17 00:12:17

现在的架构里渲染多个candidate pose的方式，非常不优雅，我认为不适合投稿，帮我分析有没有更合理有效的方式？此外，落实到与其他做camera pose refinement的工作进行对比，最终精度要达到sota级别，这也是预期目标，而且还有讨论一下初始位姿真实来源的问题

## Response:
2026/5/17 00:12:19

我会先核对仓库最新记录和相关 refinement 工作，再从架构优雅性、SOTA 对比协议、真实初值来源三方面重定目标。重点是把多候选渲染改写为可解释的概率/能量优化，而不是继续堆候选。

## Response:
2026/5/17 00:12:30

我进一步确认：多候选全渲染可保留为诊断、训练采样或上界，但主推理应转为“单初值、单渲染、迭代更新”的连续 refinement。这样也更容易把真实初值来源和 GS-CPR/GS-SMC/UGS-Loc 等 refinement baselines 公平对齐。

## Response:
2026/5/17 00:19:34

你的判断是对的：**当前“渲染多个 candidate pose，然后打分选 pose”的方式，作为诊断工具可以，作为顶刊主方法不够优雅，也很容易被审稿人归类为 sampling / lattice search。** 更严重的是，它和你最初想要的 “Pose-Observable Feature Distillation” 科学问题不完全一致：现在系统有一部分精度来自离散候选覆盖，而不是来自一个真正可微、连续、几何一致的 pose refinement 过程。

我建议下一步不要继续把多 candidate render 做得更复杂，而是把它降级为 **ablation / oracle / fallback**，主线改成：

> **render-once, solve-continuously**  
> 每次迭代只在当前 pose 渲染一次 feature/depth/world-position，然后用 POFD feature 建立 query-render correspondence，最后用几何 solver / Gauss-Newton / PnP-style update 连续优化位姿。

---

# 1. 多 candidate render 到底是不是硬伤？

不是绝对硬伤，但它不适合作为你的最终核心。

现在相关工作里确实有人用 candidate render。比如 GS-SMC 明确把初始位姿附近的 perturbed poses 渲染成 candidate images，再通过 query-render matching 和几何约束迭代 refine；它还专门讨论了 variable number of candidate images。([arXiv](https://arxiv.org/html/2508.17876v1))

这说明多 candidate render 在领域里不是不合法，但对你来说有三个问题。

第一，**创新性风险**。如果你继续保留 “render K poses -> score candidates -> choose best” 作为主路径，审稿人会自然把你和 GS-SMC / MCLoc / sampling-based refinement 放在一起比较。GS-SMC 已经把 “multiple rendered images + epipolar geometric constraints” 作为核心叙事，并声称在 7-Scenes 和 Cambridge 上相对现有 refinement 方法有显著降低 median translation / rotation error。([arXiv](https://arxiv.org/html/2508.17876v1))

第二，**科学问题不够干净**。POFD 的核心应该是：什么特征让 pose energy landscape 更适合连续优化？但多 candidate render 容易让人觉得你是在靠离散覆盖弥补 score surface 不稳定。Stage3a 里 `34x60/r12` 把 q50 从 `0.362m` 拉到 `0.225m`，这确实证明了 search radius / score path 的重要性，但它也说明系统还没有真正形成连续可优化的 pose energy。你的 Stage3a 记录也已经把剩余瓶颈定位为 hard wrong candidates 的 score ordering。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-16-pofd-stage3a-radius-scale.md))

第三，**部署和效率不优雅**。多 candidate render 的计算量随候选数线性增长；即使 3DGS 很快，K 个 pose 的 feature/depth/world-position render 仍然会让方法显得笨重。GS-CPR 这类强 baseline 已经能做到 single-query one-shot refinement，并报告 3DGS rendering 在 7-Scenes 上约 3.7ms、Cambridge 上约 12ms，整体额外开销低于 180ms。([arXiv](https://arxiv.org/html/2408.11085v4))

所以我的结论是：

> **多 candidate render 可以保留为 basin diagnosis、oracle upper bound、fallback coarse rescue；但不应该是投稿主方法。**

---

# 2. 更合理的主架构：Single-Render Continuous POFD Refinement

我建议把下一阶段命名为：

```text
Stage4: Single-Render Continuous POFD Refinement
```

或者论文里叫：

```text
POFD-GN: Pose-Observable Feature Distillation for Continuous 3DGS Pose Refinement
```

核心思想：

```text
不是渲染 K 个候选 pose；
而是在当前 pose 渲染 1 次，
用 POFD feature 建立 dense / semi-dense correspondence，
再用几何 solver 连续更新 pose。
```

---

## 2.1 主流程

当前离散候选方式大概是：

```text
T0
 ├─ render(T1)
 ├─ render(T2)
 ├─ render(T3)
 ├─ ...
 └─ render(TK)

query feature vs K rendered features
    ↓
candidate score
    ↓
select best candidate
```

建议改成：

```text
T0
 ↓
render_once(T0): feature, depth, world_position, valid_mask
 ↓
query POFD feature vs rendered POFD feature
 ↓
pair matcher predicts local correspondence / flow / confidence
 ↓
lift rendered pixels to 3D using depth/world_position
 ↓
solve ΔT with robust PnP / Gauss-Newton / WLS
 ↓
T1 = exp(Δξ) T0
 ↓
repeat 3~5 iterations
```

每一轮只渲染一次当前 pose。不是 K 个 candidate render。

---

## 2.2 为什么它更适合投稿？

因为它把方法从 “discrete search over candidates” 变成了 “continuous feature-metric pose optimization”。

论文 claim 会变得更强：

> We distill vision foundation model features into a pose-observable representation whose rendered feature residuals induce a well-conditioned continuous pose refinement objective.

中文就是：

> 我们不是从一堆候选 pose 里挑一个，而是学习一种特征，使当前 pose 附近的 query-render feature residual 能直接驱动连续 6DoF 位姿优化。

这比 “render 多个 pose 然后打分” 更像第一性原理方法。

---

# 3. 具体替代方案有三档

## 方案 A：Single-Render Correspondence + PnP / WLS

这是我最推荐的主方案。

给定当前 pose `T`，渲染：

```text
F_m(T): rendered localization feature
D_m(T): depth
X_m(T): world-position map
M_m(T): valid mask
```

query side 得到：

```text
F_q
```

pair matcher 从 `F_q` 和 `F_m(T)` 预测局部 offset：

```text
Δu_i = matcher(F_q, F_m, i)
confidence_i = c_i
```

然后每个 rendered pixel `u_i` 有一个 3D 点：

```text
X_i = X_m(u_i)
```

它在 query image 中的预测对应点为：

```text
u_i' = u_i + Δu_i
```

于是得到 2D-3D correspondence：

```text
X_i ↔ u_i'
```

最后解：

```text
T_new = argmin_T Σ_i w_i · ρ( || π(K, T X_i) - u_i' ||² )
```

这可以用：

```text
robust PnP
Gauss-Newton
Levenberg-Marquardt
weighted reprojection WLS
```

这样你的 POFD feature 不再只是 candidate score 的输入，而是直接服务于几何求解。

---

## 方案 B：Render-Once Virtual Candidate Scoring

如果你担心 q50 basin 太大，单步 continuous solver 可能收敛失败，可以保留 candidate 思想，但不再渲染多个 candidates。

做法是：

```text
只 render 当前 pose T0 一次。
```

拿到当前视角下的 3D 点 `X_i` 和 map feature `z_i`。对于一个虚拟 pose perturbation `δ`，不用重新渲染，只把同一批 3D 点重新投影到 query feature map 上：

```text
u_i(δ) = π(K, exp(δ) T0 X_i)
E(δ) = Σ_i ρ( || F_q(u_i(δ)) - z_i ||² )
```

这样可以在 tangent space 里评估一组 virtual candidates：

```text
δ1, δ2, ..., δK
```

但代价只是：

```text
一次 render + K 次 grid_sample
```

而不是：

```text
K 次 full 3DGS render
```

这可以叫：

```text
render-once trust-region pose energy
```

它比多 candidate render 更优雅，因为它明确是局部 pose energy 的 trust-region approximation，而不是 brute-force candidate search。

局限是遮挡和新可见区域处理不完美，所以它适合作为中 basin warm-start，不适合作为唯一精细 solver。

---

## 方案 C：Second-order / Taylor Pose Energy

更第一性原理的版本是直接在当前 pose 处近似 pose energy：

```text
E(T ⊕ δ) ≈ E(T) + gᵀδ + 1/2 δᵀHδ
```

其中：

```text
g = JᵀWr
H = JᵀWJ
```

`J` 是 feature residual 对 pose perturbation 的 Jacobian。

然后解：

```text
δ* = -(H + λI)^-1 g
```

这相当于 feature-metric Gauss-Newton。它是最优雅的，但实现难度和数值稳定性更高。

我建议实际推进顺序是：

```text
先做方案 A；
再加方案 B 作为 trust-region fallback；
最后再尝试方案 C 的 full feature-metric GN。
```

---

# 4. 和现有 SOTA 工作的关系

你最终要投稿顶刊，就不能只在 OldHospital 上说指标涨了。需要明确打这些 baseline。

## 4.1 GS-CPR 是必须比较的强 baseline

GS-CPR 是 ICLR 2025 的 camera pose refinement 工作，用 3DGS 渲染 RGB/depth，再用 MASt3R 建立 query-render 2D matching，最后做 pose refinement。它强调不需要训练特定 feature extractor / descriptor，而是直接使用 MASt3R。([OpenReview](https://openreview.net/forum?id=mP7uV59iJM))

它的公开结果很强：

```text
7-Scenes:
ACE + GS-CPR avg ≈ 0.8cm / 0.25°
Marepo + GS-CPR avg ≈ 0.9cm / 0.28°

Cambridge:
ACE + GS-CPR avg ≈ 15cm / 0.33°
```

这些数字来自 GS-CPR 表格。([arXiv](https://arxiv.org/html/2408.11085v4))

所以如果你要 claim SOTA camera pose refinement，至少要回答：

```text
为什么 POFD 比 “3DGS render + MASt3R matching + PnP/refinement” 更好？
```

仅仅在自有 OldHospital 上做到 `0.18m` 或 `0.20m`，不足以支撑 SOTA claim。

---

## 4.2 GS-SMC 也是强相关 baseline

GS-SMC 明确提出用 3DGS、multiple rendered images、query-render matching、epipolar geometric constraints 做 iterative pose refinement，并声称在 7-Scenes / Cambridge 上超过已有 refinement 方法。([arXiv](https://arxiv.org/html/2508.17876v1))

如果你继续保留多 candidate render 作为主方法，就会和 GS-SMC 正面撞车。你需要证明：

```text
1. 你的 POFD feature 比他们用的 generic matcher / feature 更准；
2. 你的方法更快；
3. 你的 candidate 数更少；
4. 你的连续 refinement 更稳定；
5. 你的方法在真实 init 下更强。
```

难度很大。

---

## 4.3 GSFF 是 feature-field 路线的必须比较对象

GSFF / Gaussian Splatting Feature Fields 将 3DGS 几何和 implicit feature field 结合，通过 contrastive framework 对齐 3D scale-aware feature field 和 2D encoder，并用 rendered feature / segmentation 做 pose refinement。([arXiv](https://arxiv.org/html/2507.23569v1))

如果你的论文强调：

```text
3D feature field
VFM feature distillation
rendered feature alignment
pose refinement
```

那 GSFF 必须进入相关工作和实验对比。你的区别应该是：

```text
GSFF: 学 common embedding / privacy-preserving feature or segmentation alignment
POFD: 从 VFM 中蒸馏 pose-observable feature，使 feature residual 可驱动连续 pose refinement
```

也就是说，你的核心不能只是 “我也 render feature map”，而必须是：

```text
pose-energy-aware distillation + continuous geometric refinement
```

---

# 5. 关于真实初始位姿来源

这是必须严肃处理的问题。不能继续只靠 `GT + q10/q25/q50 perturbation`。

controlled perturbation benchmark 可以保留，但它只能说明：

```text
refinement basin
score surface
oracle-free local robustness
```

它不能说明真实部署能力。

---

## 5.1 真实初始位姿应该来自哪里？

我建议至少采用三类初始化。

### 来源 A：SCR / APR 初始化

这是 camera pose refinement 论文最常用、最公平的协议。

可选方法：

```text
ACE
DSAC*
GLACE
Marepo
DFNet / PoseNet 类 APR
```

GS-CPR 就是用 DFNet、Marepo、ACE 等初始 pose，然后报告 refinement 后提升。([arXiv](https://arxiv.org/html/2408.11085v4))

这类初始化的好处是：

```text
1. 单图像；
2. 无需 retrieval oracle；
3. 和 GS-CPR / NeFeS / HR-APR / MCLoc 等工作可比；
4. refinement 的价值清楚。
```

对你的论文来说，最重要的是：

```text
同一个 initial pose，
GS-CPR refine 一次，
POFD-GN refine 一次，
比较最终误差、成功率、runtime。
```

---

### 来源 B：HLoc / retrieval + matching + PnP 初始化

HLoc 是经典 hierarchical localization pipeline，基于 image retrieval 和 feature matching，是强结构化 localization baseline。官方说明它实现了 state-of-the-art 6DoF visual localization，并且结合 retrieval、feature matching、SfM。([GitHub](https://github.com/cvg/Hierarchical-Localization))

这类初始化更接近实际视觉定位部署：

```text
query image
 ↓
image retrieval
 ↓
local feature matching
 ↓
PnP / RANSAC
 ↓
initial pose
 ↓
POFD refinement
```

但注意：HLoc 本身已经很强，refinement gain 可能很小。它适合作为：

```text
high-quality real init
```

不是最容易展示大幅提升的初始化来源。

---

### 来源 C：SLAM / VO / temporal tracking 初始化

如果你要面向 AR/robotics，可以用上一帧 pose 或 VO 作为初始位姿：

```text
T_t_init = T_{t-1} ⊕ VO_delta
```

这很真实，但需要序列数据和 temporal protocol。单图像 localization 论文里不要把它作为唯一主协议，否则会改变问题定义。

---

## 5.2 我建议的初始化协议

论文里必须分三张表。

### 表 1：Controlled basin benchmark

```text
GT + perturbation:
q10 / q25 / q50
```

用途：

```text
证明 refinement basin 和 score surface
```

不能用于 SOTA 部署 claim。

---

### 表 2：Real-init refinement benchmark

```text
Initial pose from ACE / GLACE / Marepo / HLoc
```

用途：

```text
和 GS-CPR / GS-SMC / GSFF / NeFeS / MCLoc 对比
```

这是主表。

---

### 表 3：End-to-end localization benchmark

```text
upstream initializer + POFD refinement
```

报告：

```text
median translation / rotation
success@5cm/5°
success@10cm/5°
runtime
render count
failure rate
```

这个表才能支撑 SOTA claim。

---

# 6. 我建议下一步从 candidate render pivot 到 continuous refinement

不要继续做 Stage3d PMED 作为主路线。PMED 可以保留，但我现在会把它转为辅助监督，而不是继续围绕 candidate score surface 打磨。

新的路线：

```text
Stage4a: single-render correspondence extraction
Stage4b: robust 2D-3D pose solver
Stage4c: iterative render-once refinement
Stage4d: render-once virtual candidate trust-region fallback
Stage4e: public-dataset real-init comparison
```

---

## Stage4a：让 pair matcher 输出 correspondence，而不是 candidate score

现在 pair matcher 主要服务于：

```text
candidate score
```

下一步要让它输出：

```text
offset_logits
expected_offset
offset_confidence
valid_mask
```

用于单个 rendered pose：

```text
query feature ↔ render feature
```

输出：

```text
u_render
u_query = u_render + Δu
confidence
```

不再需要 K 个 rendered candidates。

---

## Stage4b：实现 robust pose solver

输入：

```text
world_position_map X_m
predicted query correspondence u_q
confidence w
intrinsics K
current pose T
```

解：

```text
ΔT
```

目标函数：

```text
min_ΔT Σ_i w_i · ρ( || π(K, exp(ΔT) T X_i) - u_q_i ||² )
```

先实现非学习的 robust solver：

```text
Huber / Geman-McClure
LM damping
max update clamp
valid depth / alpha mask
minimum correspondence count
condition number check
```

这一步非常关键。只有 solver 起来了，你的论文才从 “feature score ranking” 变成 “camera pose refinement”。

---

## Stage4c：迭代 refinement

每轮：

```text
render_once(T_i)
match query-render
solve ΔT_i
T_{i+1} = exp(ΔT_i) T_i
```

最多：

```text
3~5 iterations
```

报告：

```text
render_count
runtime
iteration-wise error curve
failure cases
```

这会直接对齐 GS-CPR / GS-SMC / GSFF 这类 CPR 工作。

---

## Stage4d：render-once virtual trust-region fallback

对于 q50 中等扰动，如果单步 correspondence 不稳，增加一个不需要多 render 的 fallback：

```text
render current pose once
evaluate virtual δ candidates by projective sampling
choose best δ as warm-start
then run GN / PnP
```

这样你还能保留 Stage3a 的经验，但形式上不再是：

```text
render K candidate poses
```

而是：

```text
one render + local pose-energy trust-region
```

这更适合写论文。

---

# 7. 对 SOTA 精度的预期目标

你要把目标分成 internal target 和 public SOTA target。

---

## 7.1 OldHospital / 当前 POFD internal target

当前 Stage3a val128 q50 最好大约：

```text
pred_cost ≈ 0.223m
top1 ≈ 0.719
spearman ≈ 0.550
```

这是 candidate-scoring 结果，不是最终 continuous refinement。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-16-pofd-stage3a-radius-scale.md))

新的 Stage4 如果只是更优雅但精度不提升，不够。目标应该是：

```text
q50 val128:
translation median / pred_cost <= 0.18m
succ@25cm/10deg >= 0.82
oracle_gap <= 0.06m

q25:
pred_cost <= 0.10m
succ@10cm/5deg >= 0.90

q10:
pred_cost <= 0.07m
succ@10cm/5deg >= 0.93
```

更重要的是：

```text
render_count <= 5 per query
```

而不是：

```text
K candidates × render
```

---

## 7.2 Public SOTA target

如果要说 SOTA camera pose refinement，必须至少在一个公开 benchmark 上和 GS-CPR / GS-SMC / GSFF 对齐。

我建议最低目标：

### 7-Scenes

以 ACE / GLACE / Marepo 真实初始化为主：

```text
目标一：达到或超过 ACE + GS-CPR 的 avg 0.8cm / 0.25°
目标二：至少在 4/7 scenes 上优于 ACE + GS-CPR
目标三：success@5cm/5° >= 99%
目标四：runtime 不高于 GS-CPR 一个数量级
```

GS-CPR 表中 ACE+GS-CPR 在 7-Scenes 平均约 `0.8cm / 0.25°`，并且 success 表现很强。([arXiv](https://arxiv.org/html/2408.11085v4))

### Cambridge Landmarks

以 ACE / GLACE / HLoc 初始化为主：

```text
目标一：达到或超过 ACE + GS-CPR avg 15cm / 0.33°
目标二：至少在 2/4 scenes 上优于 ACE + GS-CPR
目标三：明确报告 outdoor exposure / render domain gap
```

GS-CPR 在 Cambridge 上报告 ACE+GS-CPR 平均约 `15cm / 0.33°`。([arXiv](https://arxiv.org/html/2408.11085v4))

如果你做不到这些，论文仍然可能有价值，但 claim 要变成：

```text
competitive refinement accuracy with fewer candidate renders / more interpretable pose-observable feature learning
```

而不是 SOTA。

---

# 8. 终止条件：什么时候该停止当前方向？

我建议设硬 stop rule。

## Promote 条件

Stage4 可以成为论文主方法，当且仅当满足：

```text
1. 主路径不再渲染 K 个 candidate poses；
2. 每次 refinement 最多 3~5 次 render；
3. OldHospital q50 val128 <= 0.18m；
4. q10/q25 不退化；
5. full validation 与 val128 差距不超过 15mm；
6. 至少一个真实 init 来源下，明显优于 current initializer；
7. 在至少一个公开 dataset 上接近或超过 GS-CPR / GS-SMC / GSFF 类 baseline。
```

---

## Continue 条件

继续优化 Stage4，如果：

```text
q50 val128 在 0.18m~0.20m；
q10/q25 不退化；
render_count 明显下降；
真实 init benchmark 有 >20% median error reduction；
solver failure cases 可解释。
```

这种情况说明路线正确，可以继续调 solver、confidence、visibility filter、iteration schedule。

---

## Stop / Pivot 条件

如果出现下面任一情况，就停止 POFD-continuous 主线：

```text
1. 连续 3 个 controlled config 相比 Stage3a 只提升 <10mm；
2. q50 最好仍 >0.21m；
3. continuous solver 经常发散，且 trust-region fallback 也不能稳定；
4. q10/q25 被 q50 优化明显破坏；
5. real-init benchmark 没有超过 initializer 本身；
6. public dataset 上无法接近 GS-CPR / GS-SMC；
7. 方法必须依赖 GT perturbation / GT candidate / oracle append 才能好。
```

如果触发 stop，说明 POFD feature 本身还不足以支撑 SOTA CPR，应该转向：

```text
GS-CPR-style MASt3R correspondence + POFD feature reranking / uncertainty
```

也就是不要再强行用 RADIO-derived feature 单独承担所有 correspondence。

---

# 9. 可以直接给 Codex goal mode 的任务

下面这段可以直接作为 Codex 的 goal。

```text
Goal:
Refactor the current POFD candidate-render scoring pipeline into Stage4: Single-Render Continuous POFD Pose Refinement.

Motivation:
The current architecture renders multiple candidate poses and ranks them. This is acceptable for basin diagnosis but not suitable as the final top-journal method. The final method should refine pose continuously from a real initial pose, using one render per iteration, POFD query-render correspondences, and a robust geometric solver.

Hard constraints:
- Do not use GT pose, GT camera center, GT candidate append, or oracle retrieval in deploy evaluation.
- Do not render K candidate poses as the main inference path.
- Candidate rendering may only be used for ablation, oracle upper bound, or optional fallback.
- Preserve eval_mode oracle/deploy/ablation separation.
- Keep POFD as a single localization feature, not coarse/fine retrieval.
- Do not reintroduce trainable topK selector as the main method.
- Do not use Fisher/logdet as the primary optimization target.
- Do not unfreeze 3DGS geometry in this stage.

Implementation tasks:

1. Add a single-render CPR module:
   - input: query feature, current pose T_i, camera intrinsics
   - render once at T_i
   - output rendered POFD feature, depth, alpha/valid mask, world-position map

2. Refactor PairConditionedLocalMatcher for single-render correspondence:
   - expose offset_logits
   - compute expected offset or argmax offset
   - output correspondence confidence
   - output valid correspondence mask
   - support score_feature_hw = 34x60 and 68x120

3. Implement 2D-3D robust pose solver:
   - use rendered world-position map as 3D points
   - use predicted query correspondence u_q = u_render + offset
   - solve ΔT by robust weighted reprojection minimization
   - support Huber or Geman-McClure kernel
   - support LM damping and update clamp
   - return condition number / inlier count / residual stats

4. Implement iterative refinement:
   - for i in 1..N:
       render_once(T_i)
       match query-render features
       solve ΔT_i
       T_{i+1} = exp(ΔT_i) T_i
   - default N = 3
   - max N = 5
   - report render_count and runtime

5. Add optional render-once virtual trust-region fallback:
   - render current pose once
   - evaluate local SE(3) perturbations by projective warping / grid_sample, not by re-rendering
   - use the best virtual perturbation only as warm-start
   - this must be optional and disabled in the clean main ablation

6. Training losses:
   - offset CE / GeoNCE using GT projection target
   - robust pose update loss on ΔT or final T
   - feature teacher anchor kept weak
   - candidate score losses may be retained only as auxiliary diagnostics, not main inference

7. Evaluation:
   - Controlled basin:
       q10, q25, q50 from GT perturbation
   - Real initial poses:
       ACE or GLACE init if available
       HLoc init if available
       nearest-reference-pose init only as weak baseline
   - Baselines:
       Stage3a multi-candidate scoring
       RADIO raw feature alignment
       GS-CPR-style render + MASt3R matching if feasible
       HLoc / ACE initial pose without refinement

8. Metrics:
   - median translation / rotation error
   - success@5cm/5deg, success@10cm/5deg, success@25cm/10deg
   - initial error -> final error
   - render_count
   - runtime
   - solver inlier count
   - reprojection residual
   - failure rate
   - oracle/deploy separation

Primary internal target:
- OldHospital q50 val128:
    final pred_cost or median translation <= 0.18m
    succ@25cm/10deg >= 0.82
    q10/q25 not worse than Stage3a
    render_count <= 5 per query
- Full validation:
    gap from val128 <= 15mm

Public SOTA target:
- On 7-Scenes with ACE/GLACE/Marepo initialization:
    match or improve ACE+GS-CPR average around 0.8cm / 0.25deg
    improve at least 4/7 scenes or clearly improve runtime/render efficiency
- On Cambridge with ACE/GLACE/HLoc initialization:
    match or improve ACE+GS-CPR average around 15cm / 0.33deg
    improve at least 2/4 scenes or clearly improve runtime/render efficiency

Promote condition:
Promote Stage4 as the paper main method only if:
- it removes K candidate renders from the main inference path,
- achieves OldHospital q50 <= 0.18m,
- does not regress q10/q25,
- has real-init deploy evaluation,
- and reaches competitive or better results than GS-CPR / GS-SMC-style baselines on at least one public benchmark.

Continue condition:
Continue one more tuning round if:
- q50 is between 0.18m and 0.20m,
- render_count is <= 5,
- real-init median error improves by at least 20%,
- and solver diagnostics show stable inlier/residual behavior.

Stop condition:
Write a no-go report and stop Stage4 if:
- three controlled configs improve less than 10mm over Stage3a,
- best q50 remains above 0.21m,
- continuous solver frequently diverges,
- q10/q25 regress significantly,
- or public/real-init benchmarks do not improve over the initializer or GS-CPR-style baseline.

Deliverables:
- Stage4 implementation
- configs for q10/q25/q50 controlled basin
- configs for real-init deploy evaluation
- markdown report:
  docs/superpowers/plans/2026-05-16-pofd-stage4-single-render-continuous-cpr.md
- table comparing Stage3a multi-candidate scoring vs Stage4 continuous refinement
- render_count/runtime report
- regression test summary
```

---

# 10. 最终建议

你现在的直觉非常重要：**如果最终还是渲染一堆 candidate poses 再排序，这篇论文很难显得像一个顶刊级方法。**

我建议下一步明确转向：

```text
single-render continuous pose refinement
```

把 POFD 的科学贡献落到：

```text
让 VFM-distilled localization feature 形成可连续优化的 query-render pose energy
```

而不是：

```text
让 feature 更好地区分一组离散候选 pose
```

多 candidate render 可以保留为诊断和上界，但主方法应该是：

```text
render once → match → solve ΔT → iterate
```

这条路线更优雅、更接近 camera pose refinement SOTA 工作的评价协议，也更容易和 GS-CPR、GS-SMC、GSFF 做正面对比。

---

**Sources:**

- [Camera Pose Refinement via 3D Gaussian Splatting](https://arxiv.org/html/2508.17876v1)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/refs/heads/radio/docs/superpowers/plans/2026-05-16-pofd-stage3a-radius-scale.md)
- [GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting | OpenReview](https://openreview.net/forum?id=mP7uV59iJM)
- [GitHub - cvg/Hierarchical-Localization: Visual localization made easy with hloc · GitHub](https://github.com/cvg/Hierarchical-Localization)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)