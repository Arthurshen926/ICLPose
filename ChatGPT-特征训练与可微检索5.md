# 特征训练与可微检索

**User:** Anonymous  
**Created:** 5/7/2026 20:37:27  
**Updated:** 5/12/2026 16:53:34  
**Exported:** 5/12/2026 17:06:52  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fc8787-36c8-83ea-bcd2-c2221cf5eece](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fc8787-36c8-83ea-bcd2-c2221cf5eece)  

## Prompt:
2026/5/12 13:35:30

已继续推进一轮，重点从“候选排序”转到专家建议的 **pose-conditioned energy / residual field** 主线，并做了多组新实验验证。git仓库也已更新。 

**代码推进**
- 新增两段式评估工具：[eval_pose_energy_factorized_buckets.py](/root/ICLPose/feature_extract/tools/eval_pose_energy_factorized_buckets.py)
- 修改训练脚本：[train_pose_energy.py](/root/ICLPose/feature_extract/tools/train_pose_energy.py)
  - 增加 `residual_pred_cost_m`
  - 增加 `residual_cost_gain_m`
  - 增加 `--best-metric`，可以用 residual 后的 pose cost 选 best checkpoint

**关键实验结论**
1. **translation-only + rotation noise 仍然失败**
   - 25cm/5°：selected mean 249.1mm，oracle 92.9mm
   - 50cm/10°：selected mean 499.6mm，oracle 193.7mm
   - selected identity frac：25cm 为 95.8%，50cm 为 100%
   - 结论：只要旋转没处理好，translation energy 仍会塌回中心候选。

2. **rotation-first factorized pipeline 暂时不够**
   - rotation stage 只把 5° 降到约 4.22°，10° 降到约 9.21°
   - 后续 translation stage 仍然 95.8%/100% 选择 identity
   - 结论：两段式方向合理，但当前 rotation energy 本身还不够强。

3. **identity denoising residual 也是负结果**
   - 25cm/5°：residual 后 mean 254.4mm，比 init 249.9mm 更差
   - 50cm/10°：residual 后 mean 500.2mm，基本无改善
   - 结论：单候选 `T0 -> GT` residual 回归目前不能作为主解。

**当前最重要判断**
不是候选覆盖问题，也不是 WLS step size 问题。当前瓶颈是：冻结 map/query 特征加一个浅层 PoseEnergyNet，仍然没有形成足够可靠的 pose-conditioned evidence。下一步应该停止继续调 identity denoiser / hard CE / translation-only selector，转向训练 **query-side projector / rerank adapter / feature adapter**，用 pose-energy + residual loss 真正改造 RADIO 特征空间。

**验证**
- `py_compile` 通过
- `pytest tests/test_pose_energy_net.py -q`：6 passed
- 当前无训练进程残留，GPU 已释放
- output 空间约 74GB 可用，已清理无价值失败小目录。
-----------------------------------------
目前这个项目从立项到现在尝试了很多不同的技术路线和方案，始终做不出来，要绷不住了。请先分析最新的这条路线的问题在哪里，然后从头到尾把整个项目的动机、相关工作综述、各路技术路线的方案和期望精度指标梳理一遍，不要过于乐观，到底是哪一点出了问题？

## Response:
2026/5/12 13:35:34

我先把最新 pose-energy 失败和整个项目账本合在一起复盘，不再继续堆新变体；重点是明确哪些假设已被证伪、真正卡点和可投稿边界。

## Response:
2026/5/12 13:35:49

最新证据进一步指向同一个根因：PoseEnergyNet 失败不是头部容量问题，而是当前 query-map evidence 不足以支撑可泛化 energy。接下来我会按 learning-based 约束复盘，而不是回到显式 2D-3D。

## Response:
2026/5/12 13:40:28

According to a document from 2026-05-11，我先说结论：**你现在不是“差一点调参就能成”，而是已经撞到项目最核心的假设风险了。**

最核心的问题是：

$$
\boxed{
\text{当前 RADIO/DCFF query-map feature 没有形成可泛化的 pose-conditioned evidence。}
}
$$

换句话说，候选里有好位姿，但你的模型看不出来。无论是 selector、score-map、query gate、projector、RGB L1，还是最新的 pose-energy/residual head，本质上都还在吃同一类不可靠 evidence。它们不是容量不够，而是输入证据本身对 5-25cm / 2-5° 级别的 pose 差异没有稳定排序能力。

你现在绷不住很正常，因为这不是单个 bug，而是**问题定义、特征空间、监督信号、训练闭环四件事同时没有完全对上**。

---

# 1. 最新 pose-energy / residual route 的问题在哪里

你最新这轮看起来换成了更高级的“pose-conditioned energy / residual field”，但从代码和结果看，它还没有真正跳出前面 selector 的失败原因。

最新阶段记录已经把证据写得很清楚：在 `0.25m/5° K64` val cache 上，oracle 是 `92.0mm`，说明候选池里确实有好位姿；但 `score_mean` / `score_topk_mean` 的 Spearman 只有约 `0.05`，RGB L1 的 AUC 约 `0.483`，query gate 也只从约 `252.9mm` 推到约 `250.3mm`，基本没有突破。这个结论不是“selector 没训练好”，而是“当前 topK 内没有可泛化的 pose evidence”。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-11-cpr-selector-metric-no-go-and-next.md))

最新 `PoseEnergyNet` 的设计是：输入 candidate-level score maps 和 vector priors，然后输出 energy、6D residual 和 confidence。也就是说，它吃的仍然是由 query-map local correlation 派生出来的 score-map / summary evidence。代码注释也明确说它“consumes candidate-level score maps plus vector priors”，只是比之前 selector 多预测了 residual。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_extract/students/pose_energy_net.py))

这就解释了为什么它失败：**如果 score map 本身对 pose error 不可排序，那么 energy head 和 residual head只是把不可判别证据重新包装了一遍。**

---

## 1.1 translation-only 为什么会塌回 identity

你看到：

```text
25cm/5° selected identity frac: 95.8%
50cm/10° selected identity frac: 100%
```

这不是偶然。translation-only selector 在存在 rotation noise 时，translation evidence 会被 rotation misalignment 污染。错误旋转会改变可见性、局部纹理位置、边界、遮挡和 feature alignment。此时模型很难判断“应该往哪个 translation 方向走”，于是最安全的策略就是选中心候选。

这本质上是：

$$
\text{translation evidence} \not\perp \text{rotation error}
$$

你把 translation 和 rotation 分开训，但输入 evidence 没有足够强的旋转可观测性，translation stage 自然会退化成 identity prior。

---

## 1.2 rotation-first 为什么也不够

rotation stage 只能把：

```text
5° -> 4.22°
10° -> 9.21°
```

这说明 rotation energy 也没有学到强信号。它只是在做一点点保守修正，不是识别正确 orientation。

原因还是一样：你当前 map/query feature 对小旋转差异并没有稳定的 energy landscape。RADIO 特征本身倾向语义和鲁棒性，而定位 refinement 需要对视角变化敏感。语义鲁棒和 pose-sensitive 本来就是冲突目标。

---

## 1.3 identity denoising residual 为什么是负结果

单候选 residual regression：

$$
(Q, M(T_0)) \rightarrow \Delta \xi = \log(T_{gt}T_0^{-1})
$$

看起来合理，但它要求网络能从当前 query-map mismatch 中读出**方向和幅度**。现在 local corr evidence 连“候选好坏排序”都做不到，当然更难直接回归 6D residual。

因此 residual head 会学到两种坏策略：

1. 输出接近 0，避免大错；
2. 在 train 上记忆偏移模式，val 上漂。

你看到 residual 后 mean 比 init 更差，就是这个现象。

---

## 1.4 最新 pose-energy 还不是“定位引导重建”

这是很关键的一点：你现在虽然叫 pose-energy / residual，但它并没有真正把定位 loss 回传到 map feature field。

`train_pose_energy.py` 里 candidate render 是：

```python
require_grad=False
```

并且 `_set_trainable` 只允许 `local_corr_projector.` 在 `--train-projector` 时训练；energy net 自己当然是可训练的，但 map renderer / map feature field 没有被 pose-energy loss 更新。代码还只保存 `local_corr_projector` 的 state dict。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_extract/tools/train_pose_energy.py))

所以这轮其实是：

$$
\text{frozen map/query evidence}
\rightarrow
\text{shallow PoseEnergyNet}
$$

而不是：

$$
L_{\text{pose-energy}}
\rightarrow
\text{query feature + map rerank adapter + map feature reconstruction}
$$

这点非常重要。你原始论文目标是“定位引导重建”，但当前最新 route 还没有真正做到这一点。

---

# 2. 从头梳理：这个项目最初的动机是什么

你的项目动机其实很清楚，而且是有价值的：

> 视觉定位需要既能从 3D map 中渲染，又能和 query image 对齐的特征。3DGS/2DGS 提供高效可微的新视角渲染，RADIO 提供强视觉 foundation feature。能不能把 RADIO teacher 特征蒸馏进 3DGS map，并通过定位任务反过来塑造 map/query 特征，使其服务于 camera pose refinement / relocalization？

这个动机站得住。因为 3DGS 的核心优势就是快速、高质量、visibility-aware 新视角渲染；原始 3DGS 论文也强调它使用 anisotropic splatting 和 fast visibility-aware rendering，支持高效训练和实时渲染。([Repo Sam](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/?utm_source=chatgpt.com))

RADIO / AM-RADIO 作为 teacher 也合理：AM-RADIO 是多 teacher distillation 框架，把 CLIP、DINOv2、SAM 等视觉模型的能力合到一个 backbone 中，能提供 dense visual features。([GitHub](https://github.com/nvlabs/radio?utm_source=chatgpt.com))

所以项目的核心不是错的。问题在于：

$$
\boxed{
\text{RADIO-like feature reconstruction} \neq \text{pose-sensitive localization evidence}
}
$$

你一开始把“重建强视觉特征”和“形成定位可用能量场”之间的 gap 低估了。

---

# 3. 相关工作怎么放

我建议你把相关工作分成四类，而不是混在一起讲。

---

## 3.1 传统结构式 visual localization

这类方法靠 2D-3D correspondence、PnP、RANSAC、local feature matching。它们强在几何可验证，弱在 map storage、pipeline 复杂、隐私、端到端学习不足。

你的工作不想走显式 2D-3D 主线，这可以理解。但必须承认：**视觉定位领域里最稳定的 pose evidence 仍然来自 correspondence + geometry verification**。你要做 learning-based，就必须提供同等强度的 learned evidence。

---

## 3.2 APR / SCR / learning-based relocalization

APR 直接从图像回归 pose，优点是简单快速，缺点是精度和泛化通常弱于结构式方法。LENS 这类工作已经指出，pose regression 的效果很受训练数据分布限制，并用 NeRF 新视角合成来扩充训练集，从而提升 relocalization。([arXiv](https://arxiv.org/abs/2110.06558?utm_source=chatgpt.com))

这和你的项目高度相关：你现在 val 上失败，确实像是数据分布和 pose coverage 不足。但只加 synthetic RGB 不一定够，因为你还需要 synthetic views 提供**pose-sensitive metric supervision**。

---

## 3.3 Neural feature field / render-and-refine

NeFeS 是非常相关的工作：它用 neural feature synthesis，在 test-time 渲染 dense novel-view feature，然后与 query feature 做 feature-metric refinement，增强 APR 的 3D 信息利用能力。([arXiv](https://arxiv.org/abs/2303.10087?utm_source=chatgpt.com))

你的工作可以理解成 3DGS/DCFF 版本的 feature-field CPR，但你多了 RADIO teacher 和 map/query dual feature reconstruction。问题是，NeFeS 这类方法的关键也不是单纯 feature reconstruction，而是 rendered feature 与 query feature之间能形成可优化 alignment。

---

## 3.4 3DGS-based localization / CPR

GS-CPR 用 3DGS 渲染 RGB/depth，再借助 MASt3R 建立 2D-3D correspondences，做 camera pose refinement；它本质上是“3DGS representation + strong foundation matcher + geometry”。([arXiv](https://arxiv.org/abs/2408.11085?utm_source=chatgpt.com))

SplatLoc 则把 3D Gaussian primitives 和 learned 3D descriptor fields 用于视觉定位，目标是通过精确 2D-3D matching 实现 6-DoF pose estimation。([Zju3dv](https://zju3dv.github.io/splatloc/?utm_source=chatgpt.com))

这两个工作都说明：3DGS 做 localization 是合理的，但它们都没有指望“平均 feature similarity”直接解决 pose。它们要么用强 matcher，要么用明确的 descriptor-to-geometry matching。

你这篇如果坚持 learning-based、非显式 2D-3D，就必须提出一个更强的 learned pose evidence 机制，而不是浅层 selector。

---

# 4. 你从立项到现在试过的路线

我按时间和逻辑重新梳理一下。

---

## 路线 A：完整 relocalization 全流程

目标：

$$
I_q \rightarrow \text{global retrieval/init} \rightarrow \text{render map feature} \rightarrow \text{pose refinement}
$$

期望指标：

- real-init 进入正确 basin；
- top1 或 topK candidate 能给 refiner 使用；
- 最终 single-image 6DoF relocalization 达到至少几十厘米量级，最好 <10-20cm。

当前结果：

- oracle retrieval 下 refinement 曾到约 `0.30° / 166mm`；
- 多假设 oracle-style 融合曾到约 `47.3mm`；
- 但 real-init `top1 CLS` 约 `4493.5mm`，`top10 oracle` 约 `2791.7mm`。`EXPERT_REVIEW_BRIEF_2026-05-07.md`

判断：

$$
\boxed{
\text{完整 relocalization 当前不成立。}
}
$$

不是后端完全没用，而是初始化 / candidate selection 没闭环。专家简报也早就判断，当前系统更像“定位导向特征场 + 好初始化附近较强的 pose refiner + 偏弱初始化前端”，而不是完整可部署重定位系统。`EXPERT_REVIEW_BRIEF_2026-05-07.md`

---

## 路线 B：固定初始化的 Camera Pose Refinement

目标：

$$
(I_q, \mathcal{M}, T_0) \rightarrow T^*
$$

即先不解决全局 retrieval，只研究给定初始位姿后的 refinement。

期望指标：

| bucket | 理想目标 | 可以接受的论文级目标 |
|---|---:|---:|
| 0.1m / 2° | <50mm | <80-100mm |
| 0.25m / 5° | <100mm | <120mm |
| 0.5m / 10° | <150mm | <150-200mm |
| 1.0m / 20° | 有明显 gain+ | 不能太差，最好 <300-400mm |

当前早期证据：

- 在好初始化附近，refiner 有一定能力；
- 但 WLS / learned update 不稳定；
- fixed-init 协议比 full relocalization 更符合现有证据。专家简报也建议，如果收敛论文本体，路线 B 更自然：给定初始位姿，研究 map/query 特征如何服务局部 pose refinement，强调 init-to-final gain、basin of convergence、solver robustness。`EXPERT_REVIEW_BRIEF_2026-05-07.md`

判断：

$$
\boxed{
\text{CPR 是当前最合理的论文范围，但还缺一个可靠 learned refinement 机制。}
}
$$

---

## 路线 C：coarse topK + fine-score rerank

设计：

```text
coarse branch 负责 topK recall
fine branch 负责 topK 内 rerank
```

期望：

- coarse top1 可以不稳定；
- topK 内有好 candidate；
- fine-score 能选中接近 oracle 的 candidate。

阶段性结果：

- topK / K64 candidate pool oracle 可以到 `80-92mm`；
- 说明候选池上限足够；
- 但 fine-score / selector 选不出来。

最新 no-go 结果：

```text
0.25m/5° K64 oracle: 92.0mm
delta prior: 255.3mm
feature score_mean: 390-409mm
Spearman: ~0.05
RGB L1: 415.6mm
```

这说明：

$$
\boxed{
\text{candidate coverage 足够，但 topK 内 evidence 不可排序。}
}
$$

---

## 路线 D：map-side safe-map / adapter / candidate-gradient

设计：

- safe-map / GT-align map；
- basin adapter；
- candidate-gradient 不直接改 core map。

结果：

- `adapter + safe-map` 是一轮有效改进；
- 0.5m/10° 曾到 `105.6mm`，1.0m/20° 到 `337mm`；
- 但 0.1m / 0.25m 小 basin 仍差；
- candidate-gradient 直接改 map 会把中 basin 拉坏。

判断：

$$
\boxed{
\text{safe-map/adapter 有用，但不是最终 bottleneck。}
}
$$

它改善了 map representation，但没有解决 query-map evidence 不可排序。

---

## 路线 E：trainable fine selector / score-map selector

设计：

$$
\{Q^f, M^f(T_k), \text{score maps}, \text{depth/mask stats}\}
\rightarrow
s_k
$$

期望：

- selector 能吃掉 oracle gap；
- 0.25m/5° 从 `~250mm` 拉到 `100-120mm`。

结果：

- train 可过拟合到 `120-160mm`；
- val 仍 `280-330mm`；
- soft/hard/no-coarse/score-map/query-gate 都不行；
- score-map 确认已接上，不是代码没用上。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-11-cpr-selector-metric-no-go-and-next.md))

判断：

$$
\boxed{
\text{selector 容量不是问题；输入 evidence 不泛化。}
}
$$

---

## 路线 F：query/projector metric alignment

设计：

- 训练 projector / query gate；
- 希望提升 map_q_f 和 local corr peak；
- 再回到 K64 selector。

结果：

- small perturb corr 指标可改善；
- 但 25cm K64 Spearman 仍约 `0.05`；
- query gate 只有极小改善。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-11-cpr-selector-metric-no-go-and-next.md))

判断：

$$
\boxed{
\text{局部小扰动 alignment 不能自然迁移到 K64 pose sorting。}
}
$$

这点很重要。你以为训练 local corr 会帮助候选排序，但实际没有。说明 metric 学到的是局部 patch 对齐，不是 candidate-level pose energy。

---

## 路线 G：RGB L1 / scene-coordinate head

结果：

- RGB L1 AUC `0.483`，基本随机；
- scene-coordinate head 是米级/十米级，不能作为 CPR 证据。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-11-cpr-selector-metric-no-go-and-next.md))

判断：

$$
\boxed{
\text{这些 quick geometry / appearance evidence 不能救当前主线。}
}
$$

---

## 路线 H：pose-conditioned energy / residual field

设计：

$$
(Q, M(T_k), T_k)
\rightarrow
E_k, \Delta \xi_k
$$

这是我之前建议你转的方向。理论上它比 selector 更强，因为它不仅排序，还预测 residual。

当前结果：

- translation-only + rotation noise 仍然选 identity；
- rotation-first 只能略微降 rotation；
- identity denoising residual 让 mean 更差；
- selected candidate 仍远离 oracle。

判断：

$$
\boxed{
\text{这个方向本身还没被彻底否定，但当前实现仍然在用同一套不可判别 feature evidence。}
}
$$

而且它还没有真正更新 map feature field，所以不能说“定位引导重建”已经发生。

---

# 5. 到底是哪一点出了问题？

我会把根因分成六层。越往下越根本。

---

## 5.1 第一层：你把“强视觉特征”误当成了“定位特征”

RADIO feature 很强，但它强在语义、鲁棒、多任务 dense representation。定位 refinement 需要的是：

$$
\text{small pose perturbation} \rightarrow \text{predictable feature change}
$$

也就是 pose-sensitive。

而 foundation feature 很可能学的是：

$$
\text{small viewpoint / illumination / texture change} \rightarrow \text{feature invariance}
$$

这和 pose refinement 的目标冲突。

所以根因之一是：

$$
\boxed{
\text{RADIO teacher 的 feature geometry 不等于 pose energy geometry。}
}
$$

---

## 5.2 第二层：map-side reconstruction 不是主要瓶颈，query-map metric 才是

专家简报里早就判断：map-side feature field 本身并非完全无效，问题更大概率出在 query-to-map 对齐与泛化；继续单独优化 feature reconstruction cosine，边际收益不如围绕 query-map matcher、flow confidence 和 solver 做训练闭环。`EXPERT_REVIEW_BRIEF_2026-05-07.md`

你后面的实验基本验证了这点。map 有上限，topK 有 oracle，但 query-map evidence 无法泛化排序。

---

## 5.3 第三层：candidate-level ranking 监督太稀疏

K64 selector 只告诉模型：

$$
\text{这 64 个里哪个 pose cost 更小}
$$

但没有告诉模型：

- 为什么这个 pose 好；
- 哪些局部区域贡献了 pose evidence；
- feature mismatch 的方向是什么；
- residual 应该往哪里走；
- rotation/translation 如何解耦。

这种监督很稀疏，尤其在室内重复结构、小视差、小旋转下很难学。

所以你看到：

```text
train 能过拟合
val 不能泛化
```

这是典型的“候选排序标签可以记，但 evidence 不稳定”。

---

## 5.4 第四层：你还没有真正形成 localization-guided reconstruction 闭环

最新 pose-energy 训练里，candidate render 是 no-grad，训练参数主要是 energy net 和可选 projector，而不是 map feature field。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_extract/tools/train_pose_energy.py))

所以目前主线更多是：

$$
\text{用 frozen feature 训练一个判别头}
$$

而不是：

$$
L_{\text{pose}}
\rightarrow
\text{改造 map/query feature space}
$$

这和论文目标“定位引导重建”之间有差距。

真正的定位引导重建至少要让：

$$
L_{\text{pose-energy/residual}}
\rightarrow
\text{query projector / rerank adapter / map fine decoder}
$$

但必须有强 anchor，否则又会把 map_q_f 拉坏。

---

## 5.5 第五层：rotation 和 translation 被错误地当成可轻易分解

你现在的结果说明：rotation 没处理好时，translation evidence 会塌回 identity。两段式方向可以继续研究，但当前 rotation stage 本身不够强。

所以问题不是：

$$
\text{先 rot 后 trans 就能解决}
$$

而是：

$$
\text{必须先学到 rotation-sensitive evidence}
$$

否则 translation stage 没有意义。

---

## 5.6 第六层：数据分布不够密，NVS 还没有真正用起来

3DGS 新视角扩充训练集可能有帮助，但当前代码里的 synthetic 只是 `synthetic_ratio` 下渲染 RGB，再做轻量 photometric augmentation；它没有形成大规模、pose-dense、domain-randomized、query-map metric-specific 的训练体系。`train_pose_energy.py` 里确实有 `maybe_replace_with_synthetic_queries` 和轻量 synthetic RGB augmentation，但这还不是完整的 NVS pose-energy curriculum。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/feature_extract/tools/train_pose_energy.py))

LENS 的经验很相关：新视角合成能提升 pose regression，关键原因是训练数据分布更密、更覆盖场景；他们也指出 pose regression 的准确性很大程度受训练数据规模和分布限制。([arXiv](https://arxiv.org/abs/2110.06558?utm_source=chatgpt.com))

所以 NVS 方向不是错，但你现在还没把它做到能解决问题的程度。

---

# 6. 当前项目各路线的状态表

| 路线 | 目标 | 当前证据 | 判断 |
|---|---|---|---|
| 完整 relocalization | 单图全流程 6DoF | real-init 米级 | 暂不成立 |
| fixed-init CPR | 给定 $T_0$ 后精化 | oracle / good init 有信号 | 仍是最合理论文范围 |
| coarse topK | 保证候选覆盖 | K64 oracle 80-92mm | 候选覆盖够 |
| fine-score rerank | topK 选好 pose | Spearman ~0.05 | 不成立 |
| trainable selector | 学 topK 排序 | train 过拟合，val 崩 | 不成立 |
| safe-map + adapter | 改 map feature 稳定性 | 有阶段性收益 | 可保留 |
| projector/query gate | 修 query-map metric | 小扰动改善，K64不改善 | 不够 |
| RGB L1 | 外观 rerank | 近随机 | 不成立 |
| scene-coordinate quick head | 几何证据 | 米级/十米级 | 不成立 |
| pose-energy/residual | 学 energy + residual | 仍选 identity / residual 负收益 | 当前实现不成立 |
| NVS augmentation | 扩 pose-dense 数据 | 还没完整验证 | 仍值得做，但不能只是轻量替换 RGB |

---

# 7. 不要过于乐观地说：当前投稿还差什么

如果按一篇正经 CPR / localization-oriented feature field 论文投稿，我认为至少还缺三个硬结果。

---

## 7.1 learned final 结果必须接近 oracle

现在 `0.25m/5° K64` oracle 是 `~92mm`，但 learned 还是 `250mm+`。这个 gap 太大。

最低可接受目标我会设成：

```text
0.25m/5° random K64 learned mean <= 120-140mm
0.5m/10° learned mean <= 150mm
gain+ 明显 > 0.6
```

如果继续停在 `250mm`，论文很难讲成成功方法。

---

## 7.2 需要证明定位 loss 真的改造了 feature

现在最新 pose-energy 没有真正更新 map feature field。投稿时如果 claim 是“定位引导重建”，就必须有：

```text
teacher-only feature field
vs
teacher + pose-energy feature field
```

并且证明：

```text
pose evidence Spearman / AUC 提升
CPR final error 降低
map_q_f 不崩
```

否则 reviewer 会问：你的定位引导在哪里？

---

## 7.3 需要收敛一个主线，不要再多路线并列

专家简报也指出，项目现在有两条叙事在打架：完整 coarse-to-fine localization 和 localization-oriented feature field + local pose refinement；前者要求强初始化证据，后者更符合现有结果。`EXPERT_REVIEW_BRIEF_2026-05-07.md`

现在必须砍掉一条主 claim。我的建议仍然是：

$$
\boxed{
\text{不要 claim 完整 relocalization。主线收敛到 CPR / pose-conditioned feature field。}
}
$$

---

# 8. 下一步还能怎么做，不转显式 2D-3D 的情况下

你明确说不能直接转显式 2D-3D，这个边界我接受。那么剩下的 learning-based 路线必须更激进一点，不能继续调浅层 head。

我建议只保留一个最后主攻方向：

$$
\boxed{
\text{NVS-supervised pose-conditioned feature adaptation}
}
$$

不是再训 selector，而是训练特征空间本身。

---

## 8.1 目标重新定义

不是：

$$
(Q, M(T_k)) \rightarrow \text{score}
$$

而是：

$$
(Q, M(T_k)) \rightarrow E(T_k), \Delta \xi(T_k)
$$

同时允许：

$$
L_{\text{energy/residual}}
\rightarrow
\text{query projector + rerank adapter + map fine decoder}
$$

但要加很强 anchor：

$$
L =
L_{\text{pose-energy}}
+
L_{\text{residual}}
+
L_{\text{GT-align}}
+
L_{\text{teacher}}
+
L_{\text{drift}}
+
L_{\text{variance}}
$$

---

## 8.2 NVS augmentation 必须升级

3DGS 新视角扩充训练集会有帮助，但必须满足四点：

1. **pose-dense**：围绕每个真实 train pose 采很多 `2cm-50cm / 0.5°-10°` 的 synthetic GT。
2. **query-only domain randomization**：只增强 synthetic query RGB，不增强 map render branch。
3. **real + synthetic 混训**：不能只训 synthetic，否则 domain gap 会很大。
4. **metric-specific supervision**：不仅训练 pose selector，还训练 query-map local corr、energy rank、residual direction。

否则只是多渲染一些图，帮助有限。

---

## 8.3 先做一个非常明确的 go/no-go

你现在需要止损机制。建议只再做一个大实验，不要继续无限试小变体。

### 实验名

```text
NVS-PoseEnergy-Adapter
```

### 训练参数

```text
map base frozen
rerank adapter trainable
query projector trainable
PoseEnergyNet trainable
base map decoder initially frozen
```

### loss

```text
synthetic pose-energy CE
synthetic residual Huber
real GT query-map align
real/synthetic feature drift
teacher anchor
variance/covariance anti-collapse
```

### 必看指标

```text
0.25m/5° K64 Spearman
good/bad AUC
learned selected mean
residual_pred_cost_m
map_q_f
synthetic-to-real gap
```

### go 条件

```text
Spearman > 0.25
AUC > 0.70
0.25m/5° learned mean < 150mm
map_q_f >= 0.70
```

### no-go 条件

如果这条大实验后仍然：

```text
Spearman ~0.05
learned mean >200mm
residual 不改善
map_q_f 一解冻就掉
```

那就必须承认：**在当前 RADIO/DCFF 表征和 OldHospital 数据设置下，learning-only pose evidence 这条线短期不可投稿。**

---

# 9. 最后，最诚实的判断

你问“到底是哪一点出了问题”，我的答案是：

$$
\boxed{
\text{问题出在从 RADIO/DCFF reconstruction feature 到 pose-sensitive energy feature 的中间桥没有真正建立。}
}
$$

更直白一点：

- 你已经能重建某种 map-side feature；
- 你也能训练 query student；
- 你也能生成候选池；
- 候选池里确实有好 pose；
- 但 query-map feature 的相似度、score map、RGB L1、浅层 selector、浅层 pose-energy，都不能泛化地区分哪个 pose 是好的；
- 最新 pose-energy route 也没有真正更新 map feature field，因此还不是定位引导重建。

这不是说明项目没有价值，而是说明现在的主贡献不能再写成：

> RADIO 特征重建后自然可以用于可微检索和精定位。

应该改成更准确的科学问题：

> Foundation visual features are not automatically pose-sensitive. How can a renderable 3D feature field be adapted, under pose-supervised novel-view training, into a pose-conditioned energy field for camera pose refinement?

这个问题更难，但更真实。

你现在最该做的是停止横向试小模块，把项目收缩成一个最后主线：

$$
\boxed{
\text{NVS-supervised pose-conditioned feature adaptation for CPR}
}
$$

如果它成，论文还有救，而且故事会更清楚。  
如果它也不成，就不要再硬投完整方法论文，应该降级成阶段性研究、技术报告，或者换一个更可控的子任务。

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-11-cpr-selector-metric-no-go-and-next.md)
- [3D Gaussian Splatting for Real-Time Radiance Field ...](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/?utm_source=chatgpt.com)
- [Official repository for "AM-RADIO: Reduce All Domains Into ...](https://github.com/nvlabs/radio?utm_source=chatgpt.com)
- [LENS: Localization enhanced by NeRF synthesis](https://arxiv.org/abs/2110.06558?utm_source=chatgpt.com)
- [SplatLoc: 3D Gaussian Splatting-based Visual Localization ...](https://zju3dv.github.io/splatloc/?utm_source=chatgpt.com)

## Prompt:
2026/5/12 15:45:58

如果是radio特征不符合目前定位需求的话，我们做特征域适应是否真能学出有效表征？或者是否需要更改特征类型呢？目前的这个NVS-supervised pose-conditioned feature adaptation for (CPR）整体系统架构如何设计呢？他的合理性是什么，让特征空间本身具备定位任务有效的判别性吗？

## Response:
2026/5/12 15:46:00

我先把问题拆成两层：RADIO 是否还能被适配成定位特征，以及 NVS-supervised pose-conditioned adaptation 如何形成真正闭环，而不是再做浅层 selector。

## Response:
2026/5/12 15:46:06

> **Browsing for papers and related works on radio features**
>
> [AM-RADIO: Agglomerative Vision Foundation Model ...](https://arxiv.org/html/2312.06709v5)
> [Neural Refinement for Absolute Pose Regression with Feature ...](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Neural_Refinement_for_Absolute_Pose_Regression_with_Feature_Synthesis_CVPR_2024_paper.pdf)
> [LENS: Localization enhanced by NeRF synthesis](https://proceedings.mlr.press/v164/moreau22a/moreau22a.pdf)
> [GS-CPR: Efficient Camera Pose Refinement via 3D ...](https://github.com/XRIM-Lab/GS-CPR)
> [Improved Baselines for Agglomerative Vision Foundation ...](https://www.researchgate.net/publication/386965562_RADIO_Amplified_Improved_Baselines_for_Agglomerative_Vision_Foundation_Models)
> [LENS: Localization enhanced by NeRF synthesis](https://openreview.net/forum?id=YFF73MRGUIH)
> [Neural Refinement for Absolute Pose Regression with Feature ...](https://nefes.active.vision/)
> [GS-CPR](https://xrim-lab.github.io/GS-CPR/)
> [Theia: Distilling Diverse Vision Foundation Models for ...](https://rai-inst.com/wp-content/uploads/2024/12/Theia_Distilling-Diverse-Vision-Foundation-Models-for-Robot-Learning.pdf)
> [HAL-NeRF: High Accuracy Localization Leveraging Neural ...](https://didymos-xr.eu/wp-content/uploads/2025/05/HAL-NeRF.pdf)
> [GS-CPR: EFFICIENT CAMERA POSE REFINEMENT VIA ...](https://proceedings.iclr.cc/paper_files/paper/2025/file/bf11e0bb868b6735afcdb35ff39ea831-Paper-Conference.pdf)
>
> **Considering system architecture for feature adaptation**
>
> I need to focus on system architecture for pose-conditioned feature adaptation in CPR. The core idea seems to involve using anchors carefully (like the teacher) while adding a pose-sensitive localization space. Maybe multi-feature fusion could help instead of raw RADIO. I must look at how to avoid over-optimism around feature types and invariance.
>
> **Examining pose-sensitive features and domain adaptation**
>
> I need to include pose-sensitive feature types like high-frequency RGB, gradients, and depth. Using self-supervised contrastive learning on pose perturbations could help. The teacher anchor should only provide stability. For domain adaptation, pose-dense supervision and a strong anchor are key. I must avoid shortcuts in synthetic-to-real data.
>
> **Evaluating feature types and training stages**
>
> I need to tweak the feature types—raw RADIO isn't enough. Instead of "teacher reconstruction," I'll focus on "localization" with task-specific latents. Potential features include high-res pose-sensitive, place-specific coarse features, and geometry cues like depth or gradients. Stages of training progress from frozen maps to pose-energy net training with adaptive refinements.
>
> **Addressing feature types, energy failure, and metrics**
>
> The user doesn’t want explicit 2D3D but suggests dense matching features like DINOv2 or ConvNeXt/SAM. The teacher set might shift to RADIO anchors with lower-level geometric features like DINO/MegaDepth. I'll explain why the latest energy approach failed, focusing on issues like frozen features and coupling errors. I’ll define reasonable metrics to track progress, avoiding overoptimism, and consider file citations.
>
> **Analyzing synthetic CE ranking, stages, and feature changes**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-11-cpr-selector-metric-no-go-and-next.md)
>
> **Structuring response with user summary and citations**
>
> I’m thinking it might be helpful to include a user summary without needing extra citations since we already have their text. I may browse the relevant training file, but it seems fine as-is. I’ll focus on explaining if RADIO doesn’t fit, and whether domain adaptation could work. The answer needs empathy but also directness, along with appropriate citations.
>
> **Defining feature types and NVS-supervised architecture**
>
> I’m working on choosing the appropriate feature type. Option (3) seems plausible—using a new localization feature space anchored by RADIO. The feature types would include: the stable RADIO anchor feature (not for scoring), pose-sensitive loc features, RGB residual, depth, and others as auxiliary evidence, along with uncertainty gates. For the NVS-supervised architecture, I'll define it in stages involving base maps, query encoder, and NVS generator for novel pose sampling.
>
> Thought for 4m 50s

## Response:
2026/5/12 15:50:50

According to a document from 2026-05-07，你们项目内部其实已经很早暴露了这个判断：**map-side feature field 本身不是完全无效，真正更大的问题是 query-to-map 对齐与泛化，也就是 query-map gap**；继续只优化 map-side reconstruction cosine，边际收益可能不如直接围绕 query-map matcher、confidence、solver 或 pose loss 做闭环。`EXPERT_REVIEW_BRIEF_2026-05-07.md`

我先给你一个不太乐观但可执行的判断：

$$
\boxed{
\text{RADIO 可以作为初始化和稳定 anchor，但不能继续被当成最终定位特征空间。}
}
$$

如果只是做：

$$
F_{\text{RADIO}}
\rightarrow
\text{projector}
\rightarrow
\text{selector / pose energy head}
$$

大概率还是不够。你最近的实验已经说明，冻结 RADIO/DCFF fine feature 的 score map、mean score、RGB L1、浅层 selector、浅层 pose energy 都没有形成泛化的 pose evidence。真正需要的是把特征目标从：

$$
\text{reconstruct RADIO}
$$

改成：

$$
\text{produce a pose-conditioned energy landscape}
$$

也就是让特征空间本身具备定位任务需要的**位姿敏感性 / pose-equivariance**。

---

# 1. 特征域适应到底有没有希望？

有希望，但前提很苛刻。

**浅层域适应不够。**  
只训练 projector、query gate、selector head，本质上是在问：

$$
\text{已有特征里是否已经存在 pose evidence，只是没被读出来？}
$$

你最近的 no-go 结果已经说明：至少在当前 OldHospital / K64 topK / 0.25m/5° 设置下，这个证据没有稳定存在。否则 Spearman 不会接近 0.05，selector 也不会 train 过拟合、val 崩。

**深层任务适应仍然可能。**  
真正可行的域适应应该是：

$$
L_{\text{pose}}
\rightarrow
\text{query localization feature}
$$

$$
L_{\text{pose}}
\rightarrow
\text{map localization adapter}
$$

$$
L_{\text{pose}}
\rightarrow
\text{pose-conditioned energy}
$$

也就是说，不能只是“读出 RADIO 里的信息”，而是要**改造出一个新的 localization feature space**。RADIO 只负责提供视觉先验和防止崩坏。AM-RADIO 本身是多 teacher distillation，把 CLIP、DINOv2、SAM 等模型能力融合进一个视觉基础模型；这类特征天然更偏通用视觉能力，不保证对 5cm、25cm、5° 这种 CPR 位姿扰动有足够判别性。([arXiv](https://arxiv.org/html/2312.06709v5?utm_source=chatgpt.com))

所以答案是：

$$
\boxed{
\text{可以做特征域适应，但目标不能是“更像 RADIO”，而必须是“更像 pose energy”。}
}
$$

---

# 2. 是否需要更改特征类型？

我建议不要“完全换掉 RADIO”，但必须**增加定位特征类型**。

现在你需要的不是一种 feature，而是三种互相约束的 feature：

| 特征类型 | 作用 | 是否由 RADIO 直接决定 |
|---|---|---|
| **Base visual feature** | 稳定视觉先验、语义/外观 anchor、防 collapse | 是，RADIO 适合 |
| **Localization feature** | 对位姿扰动敏感，形成可排序 energy | 不能只靠 RADIO，必须 task-learned |
| **Uncertainty / visibility feature** | 告诉模型哪些区域可信，哪些区域是遮挡、动态、低纹理、渲染伪影 | 需要从 map render + query jointly 学 |

也就是：

$$
F_{\text{map}}
=
\{F_{\text{base}}, F_{\text{loc}}, U_{\text{loc}}\}
$$

$$
F_{\text{query}}
=
\{Q_{\text{base}}, Q_{\text{loc}}, U_{\text{query}}\}
$$

其中 $F_{\text{base}}$ 继续被 RADIO anchor 约束；$F_{\text{loc}}$ 由 pose-energy / residual / NVS supervision 训练；$U$ 是可见性、置信度、区域选择。

如果你继续让一个 feature 同时承担：

```text
RADIO reconstruction
query-map cosine
topK rerank
residual regression
pose energy
```

它会互相冲突。RADIO 的强项是鲁棒语义和通用 dense 表征，而 CPR 需要的是“位姿误差导致的 feature 差异必须可预测”。这是两个不同目标。

---

# 3. NVS-supervised pose-conditioned feature adaptation 的核心合理性

这个方向的合理性不是“3DGS 能合成更多图像”这么简单，而是：

$$
\boxed{
\text{3DGS 可以提供密集、可控、带精确位姿标签的 query-map 训练对。}
}
$$

普通真实训练数据的问题是：相机位姿分布稀疏，尤其在 0.1m、0.25m、0.5m 这种 CPR bucket 中，很难覆盖足够多的局部扰动模式。NeRF/NVS 用于定位增强是有先例的：LENS 就利用 NeRF 新视角合成扩展位姿回归训练集，并指出 pose regression 受训练数据规模和分布限制很大。([arXiv](https://arxiv.org/abs/2110.06558?utm_source=chatgpt.com))

3DGS 又特别适合你，因为它能高效做 visibility-aware novel-view rendering；3DGS 原始工作强调其快速可见性渲染、各向异性 splatting 和实时渲染能力。([arXiv](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com))

你的问题是：

$$
\text{真实 query 太少，现有 RADIO feature 对 pose 不敏感}
$$

NVS 的作用是构造大量：

$$
(I_q^{syn}, T^*, \{T_k\})
$$

并且每个 $T_k$ 都有精确的：

$$
\Delta \xi_k = \log(T^*T_k^{-1})
$$

和 pose cost：

$$
c_k = d(T_k, T^*)
$$

这样你就能训练特征空间满足：

$$
E(Q, M(T^*)) < E(Q, M(T_{\text{near}})) < E(Q, M(T_{\text{far}}))
$$

并且：

$$
\Delta \xi_\theta(Q, M(T_k)) \approx \log(T^*T_k^{-1})
$$

这就是让特征空间具备定位判别性的关键。

NeFeS 的相关性在这里也很高：它不是只靠 2D image feature，而是在测试时渲染 dense feature map，并与 query feature 比较来 refine pose；这说明“rendered feature field + query feature refinement”是合理方向，但你需要把它进一步变成 **pose-conditioned task feature**，而不是仅仅重建 teacher feature。([arXiv](https://arxiv.org/abs/2303.10087?utm_source=chatgpt.com))

---

# 4. 我建议的整体系统架构

我会把系统设计成四层：

```text
Layer 1: RADIO-anchored base feature field
Layer 2: NVS-trained localization feature adapters
Layer 3: Pose-conditioned energy / residual network
Layer 4: Conservative semi-E2E localization-guided reconstruction
```

不要再做“一个 DCFF feature + 一个 selector head”的结构。那个已经被实验否掉了。

---

## 4.1 Map side：Base field + localization adapters

map-side 不要直接把 pose loss 打进全部 DCFF。应该拆成：

$$
F_{\text{map}}^{base}(T)
$$

$$
F_{\text{map}}^{loc}(T)
$$

$$
U_{\text{map}}(T)
$$

其中：

### Base feature field

$$
F_{\text{map}}^{base}
=
R_{\text{base}}(\mathcal{G}, T)
$$

它继续由 RADIO teacher / old safe-map / multi-view consistency anchor 约束。作用是保住视觉稳定性，不让特征塌。

### Localization adapter

$$
F_{\text{map}}^{loc}(T)
=
\operatorname{norm}
\left(
\operatorname{sg}(F_{\text{map}}^{base}(T))
+
\alpha A_{\text{map}}(F_{\text{map}}^{base}(T), D(T), M(T))
\right)
$$

其中：

- $A_{\text{map}}$：小 residual adapter；
- $D(T)$：rendered depth；
- $M(T)$：visibility / opacity mask；
- $\alpha$：小 residual scale，比如 0.05-0.2；
- `sg` 表示 stop-gradient，前期不要让 pose loss 直接破坏 base field。

这个 adapter 才吃 pose-energy loss。base field 前期只吃 teacher / align / drift loss。

### Uncertainty / confidence field

$$
U_{\text{map}}(T)
=
A_U(F_{\text{map}}^{base}, D, M)
$$

它不是 final confidence，而是告诉下游 energy net：

```text
哪些区域是可见的
哪些区域可能是边界/遮挡
哪些区域 feature 不稳定
哪些区域是动态或低纹理
```

这一步很重要。你现在 score_mean 不可排序，很可能一部分原因是“所有区域一视同仁”，噪声区域和有效定位区域混在一起。

---

## 4.2 Query side：RADIO anchor + localization head

query side 也要拆：

$$
Q^{base} = E_q^{base}(I_q)
$$

$$
Q^{loc} = A_q(Q^{base}, I_q)
$$

$$
U_q = A_{uq}(Q^{base}, I_q)
$$

其中 $Q^{base}$ 继续受 RADIO / old query feature anchor 约束，$Q^{loc}$ 用 pose-energy 和 residual loss 训练。

不要再直接低 LR 解冻整个 query fine。你已经看到 `map_q_f` 很快掉到 0.54，这说明 ranking / energy loss 会把 query feature 拉飞。正确做法是：

```text
freeze main query backbone
train query localization adapter
train query projector
train uncertainty head
```

也就是：

$$
Q^{loc}
=
\operatorname{norm}
\left(
\operatorname{sg}(Q^{base})
+
\beta A_q(Q^{base})
\right)
$$

前期也要小 residual scale。

---

## 4.3 Candidate generation：NVS-supervised local pose bank

训练时不要只用真实 query。每个训练 step 混合：

```text
real query batch
synthetic NVS query batch
```

### Synthetic query

采样一个 synthetic GT pose：

$$
T^*
$$

用 3DGS 渲染 query RGB：

$$
I_q^{syn} = R_{\text{RGB}}(\mathcal{G}, T^*)
$$

然后只对 query branch 做强增强：

```text
color jitter
gamma / exposure
blur / motion blur
noise
JPEG
random occlusion
crop / resize
focal / principal point jitter
```

map branch 不做这些增强。否则模型会学 renderer-to-renderer shortcut。

### Candidate bank

围绕 $T^*$ 采样候选：

$$
T_k = \exp(\xi_k)T^*
$$

候选不要只 uniform cube。要显式均衡 rotation / translation：

```text
identity candidate
translation-only candidates
rotation-only candidates
joint rotation+translation candidates
hard near candidates
far negative candidates
```

否则你最新实验里的 identity collapse 会继续出现。

---

## 4.4 Pose-conditioned EnergyNet

不要再让网络只看 summary score。它要吃三个层面的输入：

$$
Q^{loc},\quad F_{\text{map}}^{loc}(T_k),\quad \text{aux}(T_k)
$$

其中 aux 包括：

```text
depth
mask
uncertainty
candidate delta prior
noise/bucket embedding
rotation/translation perturbation type
```

输出：

$$
E_k
$$

$$
\Delta \xi_k
$$

$$
\sigma_k
$$

也就是：

```text
energy score
6D residual
uncertainty / confidence
```

我建议 EnergyNet 分三头：

### Rotation energy head

$$
E_k^R,\quad \Delta \omega_k
$$

专门判断旋转是否对。

### Translation energy head

$$
E_k^t,\quad \Delta t_k
$$

但它必须 conditioned on rotation evidence。

### Joint consistency head

$$
E_k^{joint}
$$

判断 rotation + translation 同时是否合理。

最终：

$$
E_k =
E_k^{joint}
+
\lambda_R E_k^R
+
\lambda_t E_k^t
$$

这比你现在的 translation-only / rotation-first 更稳，因为它既分解，又不完全割裂。

---

# 5. 关键 loss 设计

这是整个系统能不能从“RADIO 特征”变成“定位特征”的核心。

---

## 5.1 RADIO / old-feature anchor

防止特征崩：

$$
L_{\text{anchor}}
=
1-\cos(F^{base}_{new}, F^{base}_{old})
+
1-\cos(Q^{base}_{new}, Q^{base}_{old})
$$

以及 teacher anchor：

$$
L_{\text{teacher}}
=
1-\cos(F^{base}, F_{\text{RADIO}})
+
1-\cos(Q^{base}, Q_{\text{RADIO}})
$$

这部分只约束 base，不要强行约束 loc feature 完全等于 RADIO。

---

## 5.2 Query-map GT pose alignment

对真实和合成 query 都做：

$$
L_{\text{align}}
=
\sum_u m(u)
\left[
1-\cos
\left(
Q^{loc}(u),
F_{\text{map}}^{loc}(T^*)(u)
\right)
\right]
$$

但这个 loss 不是最终目标，只是让 query/map 进入同一个局部空间。

---

## 5.3 Pose energy ranking loss

每个 candidate 有 cost：

$$
c_k =
\|t_k-t^*\|
+
\lambda_R d_R(R_k,R^*)
$$

soft target：

$$
p_k =
\frac{\exp(-c_k/\tau)}
{\sum_j \exp(-c_j/\tau)}
$$

energy prediction：

$$
q_k =
\operatorname{softmax}(-E_k)
$$

loss：

$$
L_{\text{energy}}
=
-\sum_k p_k\log q_k
$$

这让特征空间不是“相似就好”，而是“pose cost 低，energy 必须低”。

---

## 5.4 Residual denoising loss

每个 candidate 的真实 residual：

$$
\Delta \xi^*_k
=
\log(T^*T_k^{-1})
$$

预测：

$$
\Delta \xi_k =
G_\theta(Q^{loc}, F_{\text{map}}^{loc}(T_k))
$$

loss：

$$
L_{\text{res}}
=
\rho(\Delta \xi_k-\Delta \xi^*_k)
$$

但是要注意尺度。translation 和 rotation 要分开归一：

$$
\Delta \xi =
[
\Delta t / s_t,\,
\Delta \omega / s_R
]
$$

否则 rotation 会污染 translation，或者 translation 支配 rotation。

---

## 5.5 Improvement loss

你现在 residual 后 mean 变差，说明网络没有被强制学习“更新后应该更好”。加：

$$
T'_k = \exp(\alpha\Delta \xi_k)T_k
$$

$$
L_{\text{improve}}
=
\max
\left(
0,
d(T'_k,T^*) - d(T_k,T^*) + m
\right)
$$

它的含义很直接：

```text
如果你预测 residual，至少不能让 pose 更差。
```

这比单纯 residual Huber 更贴合 CPR。

---

## 5.6 Energy monotonicity loss

更新前后 energy 应该下降：

$$
E(T'_k) < E(T_k)
$$

loss：

$$
L_{\text{mono}}
=
\max(0, E(T'_k)-E(T_k)+m_E)
$$

这会把 feature space 训练成一个可优化 energy landscape，而不是一个分类器。

---

## 5.7 Identity-collapse penalty

你现在 identity frac 过高，所以必须显式处理。

对非 identity 的正候选，如果它比 identity 更接近 GT，则要求：

$$
E(T_{\text{better}}) + m < E(T_{\text{id}})
$$

$$
L_{\text{id}}
=
\max
\left(
0,
E(T_{\text{better}})-E(T_{\text{id}})+m
\right)
$$

这直接打掉“永远选中心候选”的坏策略。

---

# 6. 训练阶段设计

不要一步端到端。建议四阶段。

---

## Stage 0：固定 safe map

使用当前最稳定的 safe-map / adapter checkpoint。

冻结：

```text
2DGS geometry
base map field
base query backbone
```

只准备 NVS 数据和 candidate bank。

---

## Stage 1：训练 localization adapters，不训 EnergyNet 大头

目标是让 $Q^{loc}$ 和 $F^{loc}$ 至少对 GT pose 对齐。

训练：

```text
query loc adapter
map loc adapter
uncertainty heads
```

loss：

```text
GT align
teacher/base anchor
old-feature drift
variance/covariance anti-collapse
small perturb contrastive
```

不要先训 K64 selector。先看：

```text
map_q_loc
local corr peak
good/bad AUC
Spearman
```

如果这一步 Spearman 仍然接近 0.05，后面 EnergyNet 大概率还是失败。

---

## Stage 2：NVS PoseEnergy pretrain

训练：

```text
EnergyNet
residual head
query loc adapter
map loc adapter
```

仍冻结：

```text
base map field
geometry
main query backbone
```

loss：

```text
energy ranking
residual denoising
improvement
monotonicity
identity-collapse penalty
anchor replay
```

synthetic:real 比例建议：

```text
synthetic 60%
real 40%
```

如果 synthetic domain gap 很大，改成：

```text
synthetic 40%
real 60%
```

---

## Stage 3：semi-E2E localization-guided reconstruction

只有 Stage 2 有明显改善后，才开：

```text
map loc adapter
map fine decoder small LR
query projector
EnergyNet
```

仍然不要开：

```text
geometry
opacity
hash MLP 全部
main query backbone
```

此时才是真正的：

$$
L_{\text{pose}}
\rightarrow
\text{feature reconstruction}
$$

但要加 drift：

$$
L_{\text{drift}}
=
1-\cos(F_{new}^{base}, F_{safe}^{base})
$$

否则 map 又会被拉坏。

---

## Stage 4：real-query fine-tuning

最后少量 real query 上 fine-tune：

```text
small LR
strong anchor
early stop by val Spearman + pose mean
```

不要只看 train pose loss。

---

# 7. 系统结构图

可以这样理解：

```text
                         3DGS / DCFF Map
                              |
                 -----------------------------
                 |                           |
          Base Feature Field           Loc Adapter
          RADIO / drift anchor         pose-energy trained
                 |                           |
          F_map_base(T)                F_map_loc(T), U_map(T)
                                             |
Synthetic / Real Query RGB                   |
          |                                  |
     Query Encoder                           |
          |                                  |
     Q_base  -------------------->  base anchor loss
          |
     Query Loc Adapter
          |
     Q_loc, U_query
          |
          |             Candidate poses {T_k}
          |                    |
          |          render F_map_loc(T_k)
          |                    |
          -------- PoseEnergyNet -------------
                          |
             E_k, Δξ_k, confidence σ_k
                          |
             select / update / rerender
```

推理时：

```text
real query
  -> Q_loc
  -> K64 candidate poses
  -> render F_map_loc(T_k)
  -> EnergyNet gives E_k and Δξ_k
  -> choose low energy candidate
  -> optionally apply 1-2 learned residual updates
```

没有显式 2D-3D/PnP/RANSAC，但仍然利用 3DGS renderable map。

---

# 8. 为什么这能让特征空间具备定位判别性？

因为监督不再是：

$$
\text{feature should look like RADIO}
$$

而是：

$$
\text{feature should make correct pose lower energy}
$$

$$
\text{feature should encode residual direction}
$$

$$
\text{feature should make update reduce pose error}
$$

具体来说，如果训练有效，应该出现三件事：

### 第一，energy 与 pose error 有 rank correlation

$$
\operatorname{Spearman}(-E_k, -c_k) \uparrow
$$

这是你现在最缺的。

### 第二，feature difference 包含 residual direction

$$
Q^{loc} - F^{loc}(T_k)
\Rightarrow
\Delta \xi_k
$$

这说明 feature 不只是“相似/不相似”，而是 pose-conditioned。

### 第三，更新后 energy 单调下降

$$
E(T'_k) < E(T_k)
$$

这才说明 feature space 像一个可优化的 pose field，而不是分类器。

---

# 9. 要不要换 feature 类型？我的实际建议

我建议分三档处理。

## 档 1：保留 RADIO，但降级为 base anchor

这是必须的。不要扔掉 RADIO，因为它提供稳定视觉先验，也能防止合成数据训练过拟合 renderer artifact。

## 档 2：新增 localization feature head

这是必须的。

$$
F^{loc} \neq F^{RADIO}
$$

它可以从 RADIO feature 初始化，但最后必须由 pose-energy / residual / NVS loss 训练。

## 档 3：增加结构型辅助特征

这不是显式 2D-3D，但可以给 learning-based EnergyNet 提供更强输入：

```text
rendered depth
visibility mask
normal / scale
edge-like high-frequency feature
uncertainty map
```

这些不等于 PnP/RANSAC；它们只是条件输入。对 CPR 来说，纯语义 feature 不够，加入结构特征是合理的。

如果这三档都做了，Spearman / AUC 仍然没有提升，那才说明需要更彻底换特征，比如引入 matching-oriented foundation feature、MASt3R/DUSt3R-style dense geometry feature，或者直接从 scratch 训练 localization feature。当前不必马上走到这一步。

---

# 10. 必须设定 go / no-go

你现在不能再无限试小模块。建议用这组指标决定这条路线是否还值得继续。

## Stage 1 go 条件

```text
0.25m/5° K64 val:
Spearman >= 0.20
AUC >= 0.65
map_q_loc >= 0.70
identity frac 明显下降
```

## Stage 2 go 条件

```text
0.25m/5° selected mean <= 160mm
0.5m/10° selected mean <= 180mm
residual_pred_cost_m < init_cost_m
residual_cost_gain_m > 0
```

## Stage 3 go 条件

```text
0.25m/5° selected mean <= 120-140mm
0.5m/10° selected mean <= 100-150mm
map_q_f / map_q_loc 不崩
```

如果 Stage 1 仍然 Spearman 约 0.05，说明 adapter 没学出定位特征；不要继续训 EnergyNet。

---

# 11. 最简实现版本

你现在最应该做的不是大重构，而是一个干净的 MVP：

```text
MVP: NVS-LocAdapter-PoseEnergy
```

只做：

1. 冻结 safe-map base。
2. 新增 query loc adapter。
3. 新增 map loc adapter。
4. 新增 EnergyNet，输出 energy + residual。
5. 用 NVS synthetic + real query 混训。
6. loss 加 energy ranking、residual、improvement、identity penalty、anchor。
7. eval 看 Spearman / AUC / selected mean / residual gain。

先不要：

```text
解冻 full map
解冻 full query
训 WLS
训 retrieval
改 coarse
改 geometry
```

---

# 12. 最终回答

**特征域适应能不能学出有效表征？**

可以，但不是现在这种浅层 projector/selector 式适应。必须把它改成：

$$
\boxed{
\text{pose-supervised, NVS-dense, residual-aware feature adaptation}
}
$$

**是否需要更改特征类型？**

需要。不是抛弃 RADIO，而是把 RADIO 从“最终特征”降级成“base anchor”，新增一个由定位 loss 训练的：

$$
\boxed{
\text{pose-equivariant localization feature}
}
$$

**NVS-supervised pose-conditioned feature adaptation 的合理性是什么？**

它利用 3DGS 的新视角能力生成密集、可控、带精确 pose residual 的训练对，让模型不再只重建 RADIO，而是学习：

$$
E(Q, M(T))
$$

和：

$$
\Delta \xi(Q, M(T))
$$

从而让特征空间本身满足：

$$
\text{正确 pose energy 低}
$$

$$
\text{错误 pose energy 高}
$$

$$
\text{feature mismatch 能预测修正方向}
$$

这才是真正的“定位任务有效判别性”。

一句话收束：

$$
\boxed{
\text{不要问 RADIO feature 能不能直接定位；要让 RADIO-initialized feature 被 NVS pose supervision 改造成 localization feature。}
}
$$

这条路仍然高风险，但它是你在“不转显式 2D-3D”的前提下，最符合论文目标、也最有可能把项目从浅层 selector no-go 中拉出来的路线。

---

**Sources:**

- [AM-RADIO: Agglomerative Vision Foundation Model ...](https://arxiv.org/html/2312.06709v5?utm_source=chatgpt.com)

## Prompt:
2026/5/12 16:47:43

那现在还有coarse、fine两种特征的必要了吗？这个Base feature field是直接对齐一个降维后的radio特征吗？即base map field、base query backbone怎么定，应该是原始radio吗？如何进一步提升方案可行性，我不想再失败了

## Response:
2026/5/12 16:47:49

我先把结论框住：不再扩展新分支，而是收敛到“稳定基座 + 定位适配特征”，重新判断 coarse/fine、base/loc 和 RADIO 的角色。

## Response:
2026/5/12 16:47:55

最新 no-go 说明当前 fine metric、RGB L1、coarse logit 都不能泛化识别 oracle，所以要把 coarse/fine 从固定分支降级为可选任务头。关键是先证明 localization feature 有 rank correlation，再谈端到端重建。

## Response:
2026/5/12 16:51:30

> **Addressing uncertainty over file citation and answer structure**
>
> The search didn’t work, and the file citation doesn't include line numbers. I should focus on the web-based citations as required, and mention the repository’s key lines. For the user’s question about coarse/fine features, I’d suggest simplifying the core tasks into base and location features, dropping the semantic/geometry distinction. Let's guide the user toward this more streamlined approach.
>
> Thought for 5m 43s

## Response:
2026/5/12 16:53:33

你现在最该做的不是继续加模块，而是**把特征体系重新收窄**。我会这样定：

$$
\boxed{
\text{旧的 coarse/fine RADIO 双分支不应再作为主线概念。}
}
$$

但这不等于完全不要多尺度。更准确的改法是：

$$
\boxed{
\text{Base visual feature}
+
\text{Localization feature}
+
\text{可选 context/detail 多尺度输入}
}
$$

也就是说，**不要再讲“coarse=语义，fine=几何”**，这个定义已经被实验削弱了。现在应该讲：

```text
Base feature：稳定视觉锚点，防止崩坏
Localization feature：由 pose-energy / residual / NVS supervision 训练出来的定位特征
Context branch：低分辨率、大感受野，辅助旋转和粗能量判断
Detail branch：中/高分辨率，辅助小位姿扰动和 residual
```

旧 coarse/fine 可以在实现里保留，但论文和主线设计里要改名、改职责。否则你会继续被旧架构绑住。

---

# 1. 现在还有 coarse / fine 两种特征的必要吗？

**有多尺度的必要，但没有旧 coarse/fine 定义的必要。**

你现在的问题不是“coarse 没训好”或“fine 没训好”，而是：

$$
\text{当前 query-map feature 没有形成 pose-sensitive energy landscape}
$$

之前的专家简报也已经指出，当前更大的问题不是单纯 map-side reconstruction，而是 query-map gap 和可泛化 correspondence 不稳定；继续只优化 reconstruction cosine，收益可能不如围绕 query-map matcher / pose loss 做闭环。`EXPERT_REVIEW_BRIEF_2026-05-07.md`

所以旧定义：

```text
coarse = semantic / retrieval
fine = geometry / correspondence
```

现在不够自洽。

我建议改成：

| 新名称 | 作用 | 是否必须 |
|---|---|---|
| `base` | 对齐 RADIO 降维特征，提供稳定视觉 anchor | 必须 |
| `loc` | 由 pose-energy / residual / NVS loss 训练，真正用于 CPR | 必须 |
| `context` | 低分辨率、大感受野，帮助旋转 / 大残差判断 | 可选，但建议保留 |
| `detail` | 中高分辨率、局部敏感，帮助 residual / 小扰动 | 可选，但建议保留 |

如果你现在资源和精力有限，MVP 可以先做：

$$
\boxed{
\text{one base feature} + \text{one loc feature}
}
$$

不要一开始再做四个分支。等 `loc` 证明有效，再把 `loc` 拆成 context/detail。

---

# 2. Base feature field 是不是直接对齐降维后的 RADIO？

**是。Base feature field 应该对齐一个冻结的、降维后的 RADIO teacher 特征。**

但有几个关键点。

## 2.1 Base 不是最终定位特征

Base 的职责不是直接拿来做 pose ranking。它的职责是：

```text
提供稳定视觉语义 / 外观 anchor
约束 map/query feature 不崩
防止 NVS synthetic 训练学到 renderer artifact
给 localization adapter 提供初始表征
```

所以 base loss 是：

$$
L_{\text{base}}
=
1-\cos(F_{\text{base}}, P_{\text{base}}(F_{\text{RADIO}}))
$$

其中 $P_{\text{base}}$ 是降维投影：

$$
P_{\text{base}}: \mathbb{R}^{D_{\text{RADIO}}} \rightarrow \mathbb{R}^{d}
$$

建议：

```text
d_base = 32 或 64
```

不要再直接存 768 / 1024 维 RADIO feature。

---

## 2.2 P_base 应该先固定，不要跟 pose loss 一起乱动

最稳做法：

1. 用训练图像提 RADIO dense feature。
2. 训练或拟合一个 `P_base`，把它降到 32/64 维。
3. 冻结 `P_base`。
4. map base field 和 query base student 都对齐这个固定空间。

也就是：

$$
Y_{\text{base}} = P_{\text{base}}(T_{\text{RADIO}}(I))
$$

map side：

$$
F_{\text{map}}^{base}(T_i)
\approx
Y_{\text{base}}(I_i)
$$

query side：

$$
Q^{base}(I_q)
\approx
Y_{\text{base}}(I_q)
$$

这个 base 空间就是你的“共同视觉参考系”。它不能被 pose-energy loss 大幅改变。

---

# 3. Base map field 和 base query backbone 应该是原始 RADIO 吗？

分两种情况。

## 3.1 训练和诊断时：必须保留 RADIO teacher upper bound

你需要一个 teacher upper-bound mode：

```text
query = frozen RADIO + P_base
map = rendered base field
```

甚至再加一个更强诊断：

```text
query = frozen RADIO(real query)
candidate = frozen RADIO(rendered candidate RGB)
```

这个诊断非常重要。它回答一个根本问题：

$$
\boxed{
\text{原始 RADIO 本身是否能区分这些 pose candidate？}
}
$$

如果：

```text
real query RADIO vs rendered candidate RADIO
```

在 K64 上 Spearman 还是接近 0.05，那说明 **RADIO 本身不是合适的 pose-sensitive feature**。这时就不要再指望单靠 projector / selector 把它救回来。

如果 teacher RADIO 有一定排序能力，但你的 DCFF rendered base field 没有，那问题在 map feature field reconstruction。

如果 teacher RADIO + rendered candidate RADIO 也不行，那必须引入 localization feature，而不是继续修 RADIO projector。

---

## 3.2 部署时：不建议直接跑原始 RADIO

原始 RADIO 可以作为 teacher，但部署侧最好是 student：

```text
query RGB
  -> query base student
  -> Q_base
  -> query loc adapter
  -> Q_loc
```

原因很现实：

- RADIO 大模型推理重；
- 你需要可训练 query localization adapter；
- 你需要和 map loc adapter 对齐；
- 部署时不应该依赖每张 query 跑巨大 teacher。

所以最终结构应是：

```text
RADIO = teacher / upper bound / anchor
query base student = deployable backbone
map base field = rendered student-compatible base space
loc adapters = 真正服务定位
```

---

# 4. 推荐的最终特征结构

我建议你把 map/query 特征都拆成两层。

## 4.1 Map side

$$
F_{\text{map}}^{base}(T)
=
R_{\text{base}}(\mathcal{G}, T)
$$

$$
F_{\text{map}}^{loc}(T)
=
\operatorname{norm}
\left(
\operatorname{sg}(F_{\text{map}}^{base}(T))
+
\alpha_m A_m(F_{\text{map}}^{base}(T), D(T), M(T))
\right)
$$

其中：

- $F_{\text{map}}^{base}$：RADIO-anchor 特征；
- $F_{\text{map}}^{loc}$：定位特征；
- $A_m$：map localization adapter；
- $D(T)$：rendered depth；
- $M(T)$：visibility / opacity mask；
- $\alpha_m$：小 residual scale；
- `sg`：stop-gradient，前期不要让 pose loss 直接破坏 base。

第一版建议：

```text
alpha_m 初始 0
最大限制 0.1 或 0.2
map base frozen
只训练 map loc adapter
```

不要再让 pose loss 直接训：

```text
hash MLP
latent
full fine decoder
geometry
opacity
```

这些自由度太大，已经多次把 basin 拉坏。

---

## 4.2 Query side

$$
Q^{base}
=
E_q^{base}(I_q)
$$

$$
Q^{loc}
=
\operatorname{norm}
\left(
\operatorname{sg}(Q^{base})
+
\alpha_q A_q(Q^{base}, I_q)
\right)
$$

其中：

- $E_q^{base}$：query base student；
- $A_q$：query localization adapter；
- $Q^{loc}$：真正进入 PoseEnergyNet 的 query 定位特征。

第一版也建议：

```text
freeze query base student
train query loc adapter
train local projector
train PoseEnergyNet
```

不要直接解冻 query fine。你已经观察到 query fine 一解冻，`map_q_f` 很快从 0.80 掉到 0.54，这说明直接用 pose/ranking loss 拉 query backbone 很危险。

---

# 5. NVS-supervised pose-conditioned feature adaptation 的完整架构

我建议你把主线固定成下面这个版本，不要再横向试很多小模块。

```text
                         3DGS / DCFF Map
                              |
                       Base map field
                    RADIO/P_base anchor
                              |
                       Map loc adapter
                              |
                     F_map_loc(T), U_map(T)
                              |
            candidate poses {T_k} around T*
                              |
                         render loc features
                              |
Query RGB / Synthetic RGB
          |
   Query base student
   RADIO/P_base anchor
          |
   Query loc adapter
          |
      Q_loc, U_q
          |
          +------ PoseEnergyNet ------+
                                      |
                         E_k, Δξ_k, confidence
                                      |
                   select / update / optional rerender
```

其中 `U_map`、`U_q` 是 uncertainty / reliability map。它们不是传统几何求解，只是告诉 EnergyNet 哪些区域可信。

---

# 6. 这个方案为什么比之前更合理？

之前失败的路线本质是：

$$
\text{固定 RADIO/DCFF feature}
\rightarrow
\text{读一个 score}
\rightarrow
\text{希望 score 能排序 pose}
$$

现在改成：

$$
L_{\text{pose-energy}}
+
L_{\text{residual}}
+
L_{\text{improvement}}
\rightarrow
A_q, A_m, E_\theta
$$

也就是不再问：

> RADIO feature 里有没有现成的定位证据？

而是训练：

> 在 RADIO base 约束下，能否学出一个 localization feature space？

特征空间的目标变成：

$$
E(Q^{loc}, F_{\text{map}}^{loc}(T_{gt}))
<
E(Q^{loc}, F_{\text{map}}^{loc}(T_{wrong}))
$$

并且：

$$
\Delta \xi_\theta(Q^{loc}, F_{\text{map}}^{loc}(T))
\approx
\log(T_{gt}T^{-1})
$$

这才叫让特征空间具备定位任务判别性。

---

# 7. 如何进一步提升可行性，避免再失败？

你现在最需要的是**减少同时变化的东西**。每一步只验证一个假设。

---

## 7.1 第一件事：做 feature upper-bound audit

在进入新一轮大训练前，先做四个 upper-bound 诊断。

| Query feature | Candidate feature | 目的 |
|---|---|---|
| RADIO teacher real query | RADIO teacher rendered candidate RGB | 测 RADIO 本身能否 pose-sort |
| RADIO teacher real query | DCFF base rendered feature | 测 map field reconstruction gap |
| query base student | RADIO teacher rendered candidate RGB | 测 query student gap |
| query base student | DCFF base rendered feature | 当前实际 base gap |

这四个实验能直接告诉你：

```text
是 RADIO 本身不适合？
是 map field 不适合？
是 query student 不适合？
还是 query-map 两边都不适合？
```

如果第一行都失败，说明不能继续把 RADIO 当定位特征，只能当 anchor。  
如果第一行有效、第四行失败，说明你的 student/map field 还有可修空间。

这个 audit 不做，后面很容易又陷入“训练失败但不知道为什么”。

---

## 7.2 第二件事：Stage 1 只训练 loc adapter，不训练 EnergyNet 大头

目标不是直接降 pose error，而是先让 evidence 变得可排序。

训练：

```text
query loc adapter
map loc adapter
local projector
```

冻结：

```text
map base
query base
geometry
PoseEnergyNet 大头可以先非常小
```

loss：

$$
L =
L_{\text{GT-align}}
+
L_{\text{small-perturb-contrast}}
+
L_{\text{anchor}}
+
L_{\text{variance}}
+
L_{\text{identity-anti-collapse}}
$$

这个阶段的成功指标不是 selected mean，而是：

```text
K64 Spearman 从 0.05 提到 >0.20
good/bad AUC >0.65
identity frac 下降
map_q_loc >=0.70
```

如果 Spearman 还是 0.05，就不要进入 Stage 2。说明 loc feature 没学出来。

---

## 7.3 第三件事：NVS 数据要做对

3DGS 新视角扩充训练集会有帮助，但不能只是“render RGB + 训练 selector”。

你需要的是：

```text
pose-dense synthetic query
+
query-only domain randomization
+
balanced candidate bank
+
energy/residual supervision
```

### Synthetic query

$$
I^{syn}_q = R_{\text{RGB}}(\mathcal{G}, T^*)
$$

对 query branch 加强增强：

```text
brightness / contrast / gamma
blur / motion blur
noise / JPEG
crop / resize
random occlusion
exposure / white balance
```

map render branch 不做同样增强。  
否则模型会学 renderer-to-renderer shortcut。

### Candidate bank 必须平衡

不要只 uniform cube。要包含：

```text
identity
translation-only
rotation-only
joint rotation+translation
near positives
hard negatives
far negatives
```

你最新 translation-only + rotation noise 塌回 identity，说明 rotation/translation 混合时模型找不到可靠 evidence。candidate bank 必须显式教它这些情况。

---

## 7.4 第四件事：EnergyNet 要分解，但不能割裂

之前 rotation-first 不够，是因为 rotation head 自己不强。translation-only 失败，是因为 rotation noise 污染 translation evidence。

新设计应该是：

```text
shared pose evidence encoder
    ├── rotation energy / residual head
    ├── translation energy / residual head
    └── joint consistency head
```

不要做完全串行：

```text
先 rot，固定后 trans
```

也不要只做：

```text
translation-only
```

公式：

$$
E_k =
E_k^{joint}
+
\lambda_R E_k^R
+
\lambda_t E_k^t
$$

residual：

$$
\Delta \xi_k =
[\Delta t_k, \Delta \omega_k]
$$

loss 分别约束 rotation / translation / joint。

---

## 7.5 第五件事：加入 anti-identity collapse loss

你已经看到 identity frac 高达 95%-100%。这必须显式惩罚。

如果某个非 identity candidate 比 identity 更接近 GT：

$$
c_j < c_{id}
$$

就要求：

$$
E_j + m < E_{id}
$$

loss：

$$
L_{\text{id}}
=
\max(0, E_j - E_{id} + m)
$$

这比指望 soft CE 自己学会更稳。

---

## 7.6 第六件事：residual 不能只看 Huber，要看 improvement

你最新 identity denoising residual 是负结果，说明单纯 residual regression 会让 pose 变差。

加：

$$
T'_k = \exp(\alpha \Delta \xi_k)T_k
$$

$$
L_{\text{improve}}
=
\max(0, d(T'_k,T_{gt}) - d(T_k,T_{gt}) + m)
$$

也就是：

```text
预测 residual 后，至少不能更差。
```

并且评价时必须看：

```text
residual_pred_cost_m
residual_cost_gain_m
identity_frac
```

这些你已经开始加了，是对的。

---

# 8. 一个更稳的训练日程

我建议你后面严格按这个顺序，不要跳。

## Stage 0：Base audit

不训练，只评估。

```text
RADIO teacher query vs RADIO rendered candidate
RADIO teacher query vs DCFF base candidate
student query vs RADIO rendered candidate
student query vs DCFF base candidate
```

目标：确定到底要不要换 feature 类型。

---

## Stage 1：Loc adapter metric shaping

训练：

```text
query loc adapter
map loc adapter
local projector
```

不训练：

```text
base map
base query
full EnergyNet
geometry
```

目标：

```text
Spearman >0.20
AUC >0.65
map_q_loc >=0.70
```

没有达到就不要继续。

---

## Stage 2：PoseEnergyNet pretrain

训练：

```text
PoseEnergyNet
query loc adapter
map loc adapter
```

loss：

```text
energy ranking
residual
improvement
anti-identity
anchor
variance
```

目标：

```text
0.25m/5° selected mean <180mm
residual_cost_gain_m >0
identity frac <50%
```

这不是最终目标，只是证明方向有效。

---

## Stage 3：NVS + real mixed training

加入大量 synthetic query，但混真实 query。

建议比例：

```text
synthetic 50%
real 50%
```

或者如果 domain gap 大：

```text
synthetic 40%
real 60%
```

目标：

```text
0.25m/5° selected mean <140-160mm
0.5m/10° selected mean <150mm
```

---

## Stage 4：低学习率 localization-guided map reconstruction

只有 Stage 3 成功后才开。

训练：

```text
map loc adapter
query loc adapter
PoseEnergyNet
可选 map fine decoder very low LR
```

不训练：

```text
geometry
opacity
full latent
full hash MLP
```

目标：

```text
证明 pose loss 真的改善 feature field
```

做 ablation：

```text
RADIO base only
+ loc adapter
+ NVS pose-energy
+ semi-E2E map loc reconstruction
```

---

# 9. 什么时候需要更换特征类型？

如果 Stage 0 发现：

```text
RADIO teacher real query vs RADIO rendered candidate
Spearman 仍接近 0.05
```

那说明 RADIO 的 dense feature 对这个 CPR pose sorting 本身就不敏感。此时不要再“相信 RADIO 只是没适应好”。

你有三个选择：

## 9.1 用 RADIO 多层特征，而不是单层最终特征

最终层可能太语义、太 invariant。可以尝试：

```text
mid-level feature
early-mid feature
final feature
multi-layer concat + projection
```

定位通常需要更局部、更结构化的中层特征，不一定是最后一层。

## 9.2 加结构辅助特征

不等于显式 2D-3D。只是作为 learning-based EnergyNet 输入：

```text
depth
normal
opacity / visibility
edge / image gradient
feature high-pass
render uncertainty
```

这能减少纯语义 feature 的歧义。

## 9.3 引入 matching-oriented teacher

如果 RADIO teacher 本身 pose-insensitive，可以考虑换或混合 teacher：

```text
RADIO base anchor
+
matching / correspondence-oriented feature teacher
+
NVS pose-energy loc feature
```

不要一开始完全抛掉 RADIO。可以让 RADIO 继续做 base anchor，loc feature 由更匹配定位的 teacher + pose loss 学出来。

---

# 10. 你现在最应该避免的事

为了减少再次失败，我建议明确禁止这些：

```text
不要再直接解冻 query fine backbone
不要再让 pose loss 打进 core map field
不要再训练单纯 selector head
不要再只看 selected mean，不看 Spearman/AUC/identity frac
不要再把 old coarse/fine 当论文主概念
不要在 Stage 1 没过时进入 Stage 2/3
```

你过去几轮最大的问题不是代码没写，而是每次都直接跑到最终指标，失败后又换头。现在必须先验证中间条件。

---

# 11. 最小可行版本应该长这样

如果你想尽量降低失败概率，MVP 就只做这个：

```text
Frozen base map field
Frozen query base student
Trainable query loc adapter
Trainable map loc adapter
Trainable PoseEnergyNet
NVS + real mixed training
Energy ranking + residual + improvement + anti-identity + anchor
```

暂时不要：

```text
coarse/fine old split
full end-to-end
retrieval
WLS
PnP
full map decoder finetune
scene-coordinate head
```

MVP 成功标准：

```text
0.25m/5° K64:
Spearman >0.20
AUC >0.65
identity frac <50%
selected mean <180mm

然后再冲 <140mm / <120mm
```

这不算保守过头。你现在 best learned 还在 250mm 左右，先把它降到 180mm，才说明方向真有信号。

---

# 12. 总结成一句话

你现在不需要保留旧的 coarse/fine 体系。应该改成：

$$
\boxed{
\text{RADIO-aligned base feature}
+
\text{pose-supervised localization adapter}
+
\text{pose-conditioned energy/residual head}
}
$$

Base map field 和 base query backbone 都应该对齐同一个**冻结的、降维后的 RADIO teacher 空间**，但它们不是最终定位特征。最终用于定位的是：

$$
F^{loc}, Q^{loc}
$$

它们必须通过 NVS pose supervision、energy ranking、residual improvement 和 anti-identity loss 学出来。

最关键的是：**先做 RADIO teacher upper-bound audit。**  
如果原始 RADIO 自己都不能区分 candidate pose，那就别再指望浅层 projector 救它；RADIO 只能做 base anchor，loc feature 必须换成或学习成更 pose-sensitive 的特征。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)