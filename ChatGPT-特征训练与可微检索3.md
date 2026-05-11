# 特征训练与可微检索

**User:** Anonymous  
**Created:** 5/7/2026 20:37:27  
**Updated:** 5/11/2026 12:48:23  
**Exported:** 5/11/2026 12:48:55  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fc8787-36c8-83ea-bcd2-c2221cf5eece](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fc8787-36c8-83ea-bcd2-c2221cf5eece)  

## Response:
2026/5/11 12:27:22

我会把这轮当作路线复盘：重点核查 selector 泛化失败是否源于 query-map metric 未对齐，而不是候选池或容量问题。接下来会对照最新实现，判断最短收敛路线与投稿风险。

## Response:
2026/5/11 12:27:27

最新实现已接入 fine selector、random/jitter eval 和缓存训练，因此失败不在缺模块；关键证据是 train 可过拟合、val 崩，说明 frozen fine metric 本身不可泛化排序。

## Response:
2026/5/11 12:31:11

> **Investigating potential selector head bug**
>
> I think there might be an issue where the "checkpoint warmstart" doesn’t include the selector head when loading a frozen selector. The key problem could be that `project_query_render_for_fine_selector` only calls `local_corr_projector`, not `LocalCorrDomainAdapter`. The code skips over using `local_corr_domain_adapter`, which feels like a bug if it’s defined separately in the model.
>
> Thought for 5m 23s

## Response:
2026/5/11 12:32:46

你这轮结论很关键，而且我会直接说：**现在确实离最早的投稿目标还有距离，但方向已经比之前清楚很多。** 你已经把问题从“是不是 coarse/topK 不够”推进到了更本质的一层：

$$
\boxed{
\text{query-map fine metric 本身在未见样本上不可排序}
}
$$

所以你的下一步判断是对的：**先做强 anchor 的 query/projector metric 对齐，再回到 K64 selector。**  
不要继续直接训 K64 selector，也不要回到 coarse/WLS/map-gradient。

---

## 1. 现在真正发生了什么

你现在的证据链是：

$$
\text{uniform K64 oracle} \approx 80.6\text{mm}
$$

说明候选池上限已经达标。

但：

$$
\text{learned selector val} \approx 244\text{mm+}
$$

说明 selector 没法从当前 evidence 中泛化地识别好候选。

再加上：

$$
\text{train 可过拟合到 }120\text{--}160\text{mm}
$$

$$
\text{val 仍然 }280\text{--}330\text{mm}
$$

这基本排除了“selector 容量不够”的解释。更像是：

$$
\boxed{
\text{当前 frozen query/map fine correlation 在 train 内有可记忆信号，但在 val 上不是稳定 pose evidence}
}
$$

你之前的专家简报其实已经预示了这个方向：项目瓶颈很可能不是 map-side reconstruction 本身，而是 query-map gap，也就是 query student / map feature 之间的可泛化 correspondence 还不稳定。`EXPERT_REVIEW_BRIEF_2026-05-07.md`

---

## 2. 这轮已经排除掉的路线

### 2.1 不要再降 K

这个已经明确了：

| candidate pool | oracle mean |
|---|---:|
| K64 | 80.6mm |
| K32 | 112.5mm |
| K16 | 152.8mm |

所以 K64 不是浪费。K32/K16 会直接损失上限。现在不该追“更小 K 更快”，应该接受：

$$
\boxed{
K64 \text{ 是当前小 basin 的合理候选规模}
}
$$

之后可以做 coarse prefilter，但现在不要牺牲 K64 上限。

---

### 2.2 不要继续调 frozen selector

你已经试了：

- cached frozen selector；
- no-coarse selector；
- hard/soft variant；
- score-map cache；
- feature column ablation；
- hard/soft target。

仍然 val 失败。这说明 current evidence 不是一个简单 MLP/Transformer 能泛化排序的空间。

仓库里的 `train_fine_selector_cache.py` 现在确实是冻结模型、只训练 `fine_candidate_selector_head`：代码先加载模型和 selector，然后把整个 model 参数都 `requires_grad_(False)`，只给 selector 参数打开梯度。([GitHub](https://github.com/Arthurshen926/ICLPose/blob/radio/feature_extract/tools/train_fine_selector_cache.py)) 这条线失败后，就不能继续怪 selector 容量。

---

### 2.3 不要只解冻 projector

你已经验证：

$$
\text{online projector+selector soft val} \approx 308\text{mm}
$$

这说明单靠 projector 被 selector loss 拉，也不够。原因是 K64 selector loss 太稀疏、太离散，容易让 projector 学到 candidate-set shortcut，而不是 dense query-map metric。

---

### 2.4 绝对不要直接解冻 query fine

你看到：

$$
map\_q\_f: 0.80 \rightarrow 0.54
$$

这很危险。它说明 query fine 被 K64 ranking loss 拉离了 map feature space。这个现象比 selector val 差更重要，因为它说明：

$$
\boxed{
\text{ranking loss 对 query fine 来说太弱约束、太不稳定，不能单独作为 metric-learning 主损失}
}
$$

如果继续直接解冻 query fine，最后会变成“selector train 上更会记忆，val 上 query-map 对齐更坏”。

---

## 3. 代码层面还有一个必须检查的点

我看到 `train_fine_selector_cache.py` 支持 `score_maps`，但它是有条件使用的：cache 里有 `score_maps` 时才传入 loss；如果 selector 的 `expects_score_map` 为真但 cache 不含 score map，会报错。([GitHub](https://github.com/Arthurshen926/ICLPose/blob/radio/feature_extract/tools/train_fine_selector_cache.py)) 训练入口也只拿 `dataset.tensors[0:4]` 作为 vector evidence，并把第 5 个 tensor 当作 `score_maps` 传入 loss。([GitHub](https://github.com/Arthurshen926/ICLPose/blob/radio/feature_extract/tools/train_fine_selector_cache.py))

你需要马上确认一件事：

$$
\boxed{
\text{当前失败的 selector 是否真的消费了 score-map，而不是只消费 vector stats？}
}
$$

因为脚本 docstring 仍写的是 “vector-only fine candidate selector”。([GitHub](https://github.com/Arthurshen926/ICLPose/blob/radio/feature_extract/tools/train_fine_selector_cache.py)) 如果 config 里的 selector `expects_score_map=False`，那你导出的 score-map cache 实际上可能没有发挥作用，或者只作为无效字段被带着走。

建议加一个硬日志：

```python
print({
    "selector_class": selector.__class__.__name__,
    "expects_score_map": bool(getattr(selector, "expects_score_map", False)),
    "has_score_maps": score_maps_b is not None,
    "score_maps_shape": None if score_maps_b is None else tuple(score_maps_b.shape),
})
```

再加一个回归测试：

```python
# 当 score_maps 被随机打乱时，val metric 应该明显变化；
# 如果不变化，说明 selector 没真正使用 score_maps。
```

如果 score-map selector 没真正吃 map，那这轮“score-map selector 失败”的结论要打折。  
如果确实吃了 score-map仍失败，那就更支持 metric alignment 阶段。

---

# 4. 我对当前瓶颈的判断

现在瓶颈不是：

$$
\text{candidate coverage}
$$

也不是：

$$
\text{selector capacity}
$$

也不是：

$$
\text{WLS update scale}
$$

而是：

$$
\boxed{
\text{query fine 与 rendered map fine 的局部相关性没有形成可泛化的 pose-energy landscape}
}
$$

更具体地说，当前 local corr 可能在 train query 上能被 selector 学会一些模式，但在 val query 上：

1. 好 candidate 和坏 candidate 的 score-map 分布不可分；
2. score peak 位置、peak margin、mask coverage 和真实 pose error 的相关性弱；
3. query/map feature 的投影空间对不同图像、不同视角、不同纹理区域没有稳定校准；
4. K64 内候选相似度太高，手工或学习 selector 都容易被局部纹理/mask/重复结构误导；
5. query fine 一旦用 ranking loss 解冻，会迅速偏离 map feature anchor。

所以你下一步说的“强 anchor 的 query/projector metric 对齐阶段”是对的。但要注意：**这个阶段不能再以 K64 selector loss 为主损失。**

---

# 5. 下一步主线：Metric Alignment Before Selection

我建议把下一阶段命名为：

$$
\boxed{
\text{Phase6-A: Strong-Anchored Query-Map Metric Alignment}
}
$$

目标不是直接降低 K64 selector error，而是先让下面三个指标变好：

$$
map\_q\_f \ge 0.70
$$

$$
\text{local corr peak / margin 提升}
$$

$$
\text{pose-error 与 fine-corr evidence 的 rank correlation 提升}
$$

然后才进入：

$$
\boxed{
\text{Phase6-B: K64 Fine Selector}
}
$$

---

## 6. Phase6-A 具体怎么做

### 6.1 冻结 map，训练 query/projector，不训练 selector

第一阶段只训练：

```text
local_corr_query_projector
local_corr_render_projector
query_fine_adapter / query projection head
```

但冻结：

```text
map renderer
map fine decoder
candidate scorer
fine selector
main query backbone 大部分参数
```

如果你要解冻 query fine，也只能解冻最后的 projection/adapter，不能解冻整个 fine head。

推荐参数：

```yaml
train:
  map_renderer: false
  candidate_scorer: false
  fine_selector: false
  local_corr_projector: true
  query_fine_adapter: true
  query_backbone: false
```

学习率：

```yaml
local_corr_projector_lr: 1e-4
query_fine_adapter_lr: 1e-5 ~ 3e-5
main_query_backbone_lr: 0
```

---

### 6.2 用 GT pose dense alignment 做主损失

不要用 K64 ranking 做主损失。先用 GT pose 渲染 map fine：

$$
M^f_{gt} = R_f(\mathcal{M}, T_{gt})
$$

query fine：

$$
Q^f = E_q(I_q)
$$

project 后：

$$
\tilde{Q}^f = P_q(Q^f)
$$

$$
\tilde{M}^f_{gt} = P_m(M^f_{gt})
$$

主 anchor：

$$
L_{\text{align}}
=
\sum_u m(u)
\left[
1 -
\cos
\left(
\tilde{Q}^f(u),
\tilde{M}^f_{gt}(u)
\right)
\right]
$$

这个 loss 的目标就是守住：

$$
map\_q\_f
$$

你已经看到 query fine 会漂，所以这个 loss 权重要大。

建议：

```yaml
gt_pose_query_map_cos_weight: 1.0
```

---

### 6.3 加 local correlation CE / peak loss

只做 cosine 不够，因为你最终靠 local corr 排候选。你需要显式训练：

$$
\text{GT pose 下 local corr peak 应该在 zero offset}
$$

对每个 pixel $u$，在 rendered GT feature 的局部窗口内做 correlation：

$$
C_u(\Delta)
=
\tilde{Q}^f(u)^\top
\tilde{M}^f_{gt}(u+\Delta)
$$

正样本：

$$
\Delta = 0
$$

loss：

$$
L_{\text{corr-ce}}
=
-\log
\frac{\exp(C_u(0)/\tau)}
{\sum_{\Delta \in \mathcal{N}} \exp(C_u(\Delta)/\tau)}
$$

再加 peak margin：

$$
L_{\text{margin}}
=
\max
\left(
0,
m -
C_u(0)
+
\max_{\Delta \ne 0} C_u(\Delta)
\right)
$$

这比 K64 selector loss更直接，因为它训练的是局部 metric，而不是候选排序。

建议：

```yaml
local_corr_ce_weight: 1.0
local_corr_peak_margin_weight: 0.2
local_corr_radius: 4
local_corr_temperature: 0.05
```

---

### 6.4 加 small perturbation contrastive，不要上 K64

GT pose 对齐只告诉模型“正确视角应该相似”。还要告诉它“轻微错误视角应该可区分”。

采样小扰动：

$$
T^- = \exp(\delta \xi)T_{gt}
$$

例如：

```text
translation: 2cm, 5cm, 10cm
rotation: 0.5°, 1°, 2°
```

渲染：

$$
M^f_{-} = R_f(\mathcal{M}, T^-)
$$

做 pose-aware contrastive：

$$
L_{\text{pose-contrast}}
=
-\log
\frac{
\exp(s(Q,M_{gt})/\tau)
}{
\exp(s(Q,M_{gt})/\tau)
+
\sum_j \exp(s(Q,M^-_j)/\tau)
}
$$

这里的 $s$ 不要只用 global mean cosine，最好用 local corr peak 或 masked top-k corr。

这一步目标是让 metric 对 2-10cm 级别的 pose 差异有响应。当前 0.25m/5° 小 basin selector失败，很可能就是因为候选之间太相似，metric 没有形成足够的局部斜率。

建议：

```yaml
small_perturb_contrast_weight: 0.2
perturb_trans_cm_choices: [2, 5, 10]
perturb_rot_deg_choices: [0.5, 1.0, 2.0]
```

---

### 6.5 加 RADIO / old-query drift anchor

为了防止 query fine 再次从 0.80 掉到 0.54，需要保守 anchor。

假设原始 query fine 是：

$$
Q^f_0
$$

训练后的 query fine/projected fine 是：

$$
Q^f
$$

加：

$$
L_{\text{query-drift}}
=
1 -
\cos
\left(
Q^f,
\operatorname{sg}(Q^f_0)
\right)
$$

再加 teacher anchor：

$$
L_{\text{teacher}}
=
1 -
\cos
\left(
Q^f,
P(T_{\text{RADIO}}(I_q))
\right)
$$

建议先用 drift，而不是强 teacher。因为你的最终 metric 是 query-map，不是纯 RADIO teacher reconstruction。

```yaml
query_old_feature_drift_weight: 0.5
query_teacher_anchor_weight: 0.1
map_q_f_min_guard: 0.70
```

一旦 val `map_q_f < 0.70`，直接 early stop 或降低 LR。

---

### 6.6 加 variance / covariance 防 collapse

如果用大量 cosine/CE，projection 可能塌缩。加 VICReg 风格约束：

$$
L_{\text{var}}
=
\sum_c \max(0, \gamma - \sigma_c)
$$

$$
L_{\text{cov}}
=
\sum_{i \ne j} \operatorname{Cov}_{ij}^2
$$

建议小权重：

```yaml
feature_variance_weight: 0.05
feature_covariance_weight: 0.01
```

---

## 7. Phase6-A 的总 loss

我建议先这样：

$$
L =
1.0 L_{\text{GT-align}}
+
1.0 L_{\text{corr-CE}}
+
0.2 L_{\text{peak-margin}}
+
0.2 L_{\text{small-perturb-contrast}}
+
0.5 L_{\text{query-drift}}
+
0.1 L_{\text{teacher-anchor}}
+
0.05 L_{\text{var}}
+
0.01 L_{\text{cov}}
$$

暂时不要加：

$$
L_{\text{K64-selector}}
$$

或者最多加极弱：

```yaml
fine_topk_selector_weight: 0.02
```

但我建议第一轮完全不加，先看 metric 是否变好。

---

# 8. Phase6-A 该看什么指标

不要只看 final pose。这个阶段先看 metric。

每次 val 报：

```text
map_q_f_cos
local_corr_peak_acc@r0
local_corr_peak_acc@r1
local_corr_peak_margin
local_corr_entropy
fine_score_vs_pose_error_spearman
good_bad_candidate_auc
K64_oracle_mean
K64_fine_score_mean
```

其中最重要的是：

### 8.1 Spearman / Kendall

对每个 query 的 K64 candidate，算：

$$
\rho =
\operatorname{Spearman}
\left(
-\text{pose error}_k,
\text{fine evidence score}_k
\right)
$$

如果 metric alignment 真的有效，val 上这个 $\rho$ 应该上升。  
如果 $\rho$ 仍接近 0，selector 继续训也没用。

### 8.2 Good/bad candidate AUC

定义：

```text
good: trans_err < 0.10m 或 top 10% error
bad:  trans_err > 0.25m 或 bottom 50%
```

用 fine evidence 做 AUC。  
如果 AUC 不过 0.65-0.70，selector 很难到 80mm oracle。

### 8.3 gap 分解

对 K64：

```text
oracle mean
best fine evidence mean
learned selector mean
```

Phase6-A 目标不是 learned selector 立即变好，而是：

```text
best fine evidence / simple fine score 从 244mm 降到 150mm 左右
```

如果 metric 本身变得可排序，后面 selector 才有意义。

---

# 9. Phase6-B：再训 K64 selector

等 Phase6-A 达到：

```text
map_q_f >= 0.70
Spearman/AUC 明显提升
local corr peak/margin 提升
```

再回到 selector。

这时建议：

### 9.1 冻结 metric，只训 selector

```yaml
train:
  query_projector: false
  query_adapter: false
  selector: true
```

### 9.2 用多 bank cache，不要单 bank

你现在 train 能过拟合、val 不行，一个高概率问题是：selector 记住了固定候选池分布。

建议每个 query 导出多个 K64 pool：

```text
bank0: uniform cube K64 seed A
bank1: uniform cube K64 seed B
bank2: rank_uniform
bank3: random jittered lattice
bank4: mixed small/medium perturb
```

训练时同一个 query 会见到多个 candidate set。这样 selector 学的是 evidence，不是候选位置模式。

### 9.3 用 bucket-balanced training

不要只训 0.25m/5°。建议混合：

```text
0.1m/2°:   30%
0.25m/5°: 40%
0.5m/10°: 20%
1.0m/20°: 10%
```

因为你的小 basin 是投稿目标短板。

### 9.4 目标仍然用 soft pose-error label

$$
p_k =
\operatorname{softmax}
\left(
-\frac{
e^t_k + \lambda_R e^R_k
}{\tau}
\right)
$$

但温度分 bucket：

```text
0.1m/2°:   tau = 0.03m
0.25m/5°: tau = 0.05m
0.5m/10°: tau = 0.08m
1.0m/20°: tau = 0.15m
```

---

# 10. 你现在要避免的三个误区

## 10.1 不要把“selector 失败”理解成“selector 还不够复杂”

当前证据更像 metric 不可排序。继续把 selector 从 MLP 换成 Transformer、CNN、attention，大概率只会让 train 更好、val 更差。

## 10.2 不要用 K64 loss 直接拉 query fine

你已经看到 query fine 漂了。K64 ranking 是下游稀疏监督，不足以定义 dense metric space。必须先用 GT pose dense alignment / local corr CE / drift anchor 稳住。

## 10.3 不要回到 full localization claim

这轮结果进一步支持：当前论文主线仍应是 CPR / feature-field refinement，而不是完整单图重定位。仓库 README 也仍把 real deployment gap 指向 candidate scoring / selector quality，而不是下游 solver 单独问题。([GitHub](https://github.com/Arthurshen926/ICLPose/tree/radio))

---

# 11. 对投稿目标的现实判断

我会很直接地说：**如果目标是近期投稿，现在还不能把 claim 写成“完整 visual localization / relocalization”。**

更现实的目标是：

$$
\boxed{
\text{Camera Pose Refinement with Task-Aligned Query-Map Feature Fields}
}
$$

但即便是 CPR，当前结果也还缺一个硬点：

$$
0.25m/5^\circ \text{ learned final mean } < 100\text{--}120\text{mm}
$$

你现在 oracle 是 80mm，说明这不是不可能；但 learned selector 244mm+ 说明目前主线还没吃到上限。

我建议给自己设一个明确 go/no-go：

### Go 条件

在 random cube K64 / jittered eval 下：

```text
0.25m/5° learned selector mean <= 120mm
0.5m/10° learned selector mean <= 80mm
map_q_f >= 0.70
selector train-val gap <= 50mm
```

达到这个，论文可以继续按 CPR 投。

### No-go 条件

如果 Phase6-A 后：

```text
map_q_f 能守住，但 Spearman/AUC 不上升；
或者 learned selector 仍 >180mm；
```

那说明 RADIO-style dense feature 对这个 K64 CPR sorting 任务不够，短期内要换论文侧重点：少讲 learned selector，改讲 map feature field + oracle/local candidate upper bound + failure analysis 不够强，或者把目标延后。

---

# 12. 我建议你下一步只做四件事

## 第一件：验证 score-map selector 是否真的用了 score map

这是最小检查。不要跳过。

```text
打乱 score_maps，如果 val 不变，说明 score-map 没接上或没被用。
```

## 第二件：做 Phase6-A metric alignment

冻结 map，训练 projector/query adapter，强 anchor：

```text
GT query-map cosine
local corr CE
peak margin
small perturb contrastive
query drift
teacher anchor
variance/covariance
```

目标：

```text
map_q_f >= 0.70
local corr peak/margin 上升
Spearman/AUC 上升
```

## 第三件：导出 multi-bank K64 cache

每个 query 多个候选池，避免 selector 记 fixed candidate pattern。

## 第四件：再训 selector

只在 metric 变好后做。否则 selector 继续失败是预期内的。

---

## 最终结论

你这轮不是“白忙了”，而是把问题定位到了更深的一层：

$$
\boxed{
\text{当前差距不是候选池，不是 selector 容量，而是 query-map fine metric 在 val 上不可排序。}
}
$$

所以下一步路线应该改成：

$$
\boxed{
\text{强 anchor metric alignment}
\rightarrow
\text{multi-bank K64 selector}
\rightarrow
\text{再看是否能冲 }0.25m/5^\circ < 100\text{--}120\text{mm}
}
$$

你最早的投稿目标还没到，但现在至少知道该往哪里打：**先让 query-map fine correlation 成为可泛化的 pose evidence，再让 selector 学排序。**

---

**Sources:**

- [ICLPose/feature_extract/tools/train_fine_selector_cache.py at radio · Arthurshen926/ICLPose · GitHub](https://github.com/Arthurshen926/ICLPose/blob/radio/feature_extract/tools/train_fine_selector_cache.py)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)