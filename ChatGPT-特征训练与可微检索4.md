# 特征训练与可微检索

**User:** Anonymous  
**Created:** 5/7/2026 20:37:27  
**Updated:** 5/11/2026 20:46:17  
**Exported:** 5/11/2026 22:13:37  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fc8787-36c8-83ea-bcd2-c2221cf5eece](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fc8787-36c8-83ea-bcd2-c2221cf5eece)  

## Prompt:
2026/5/11 20:40:46

首先不能直接转向显式2d3d方式，这个绝对不能变，本文就是要探讨端到端learning-based方法，而且是定位引导重建。再想想其他解决方案，这条技术路线不可能完全不通。此外，我如果利用3dgs新视角的特性来扩充训练集，是否会好一些？

## Response:
2026/5/11 20:40:49

我会先固定边界：不转显式 2D-3D/PnP，也不再微调已 no-go 的 selector 变体；重点改为学习式 pose-energy 与 3DGS 新视角增强。

## Response:
2026/5/11 20:40:55

当前被否定的是特征相似度/手工 correlation 作为排序证据，而不是 learning-based 路线；3DGS 扩增应补成密集、可控的 pose-energy 预训练。

## Response:
2026/5/11 20:46:17

你这个边界我接受：**不能转显式 2D-3D / PnP-RANSAC 主线**。那我前一轮把它作为主方向提出来，不适合你这篇文章的核心目标。你的目标是 **end-to-end learning-based、定位引导重建、query-map feature learning**，那就不能把问题甩给传统几何验证。

但这不等于现在这条技术路线不通。更准确的判断应该是：

$$
\boxed{
\text{“手工/浅层 fine similarity → topK selector”这条子路线不通，}
}
$$

而不是：

$$
\boxed{
\text{learning-based CPR / 定位引导特征场不通。}
}
$$

你现在需要换的不是研究路线，而是**学习信号的形式**。

---

# 1. 现在到底哪里错了

你最新总结里的 no-go 其实很清楚：

| evidence | 结果 |
|---|---:|
| K64 oracle | 约 `92mm` |
| delta prior | 约 `255mm` |
| feature score_mean | 约 `390-409mm` |
| Spearman | 约 `0.05` |
| RGB L1 | 约 `415mm` |
| query gate | 基本没改善 |
| scene-coordinate head | 米级 / 十米级 |

这说明：

$$
\text{K64 里有好 pose}
$$

但：

$$
\text{当前 query-map fine feature 不能泛化地区分这些 pose}
$$

最新阶段记录也写得很直接：K64/topK 内有好候选，但当前 query-map fine metric、RGB L1、coarse logit 都不能泛化识别 oracle；delta prior 只能保 basin，远达不到目标。([GitHub](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-11-cpr-selector-metric-no-go-and-next.md))

所以我现在会把问题重新定义成：

$$
\boxed{
\text{当前模型没有学到一个 pose-conditioned energy landscape。}
}
$$

你之前一直在做的是：

$$
(Q, M(T_k)) \rightarrow s_k
$$

然后希望 $s_k$ 能排序。

但真正需要的是：

$$
(Q, M(T), T) \rightarrow E(Q,T)
$$

其中 $E(Q,T)$ 应该是一个**连续、可学习、可微的位姿能量函数**：

$$
E(Q,T_{\text{gt}}) < E(Q,T_{\text{near}}) < E(Q,T_{\text{far}})
$$

并且它最好还能输出一个 pose residual：

$$
\Delta \xi_\theta(Q, R_{\mathcal{M}}(T)) \approx \log(T_{\text{gt}}T^{-1})
$$

这才是 learning-based CPR 的核心，而不是继续调一个 topK 内的浅层 selector。

---

# 2. 当前架构还是不是最初那套？

**概念上还是，但实现主线已经偏了。**

最初你要的是：

```text
RADIO teacher
  -> map-side feature field
  -> query-side feature
  -> differentiable retrieval / refinement
  -> localization-guided reconstruction
```

这个大方向仍然成立。

但这几轮实际做成了：

```text
safe-map / adapter
  -> K64 candidate pool
  -> feature score map
  -> frozen selector / trainable selector / query gate / projector
  -> no-go
```

这条子路线的问题是：它把 learning-based localization 简化成了一个 **candidate sorting problem**。而 candidate sorting 的监督太稀疏，输入 evidence 又没有被训练成 pose-sensitive，所以 selector train 能过拟合，val 完全不行。

你的专家简报里早就有一个关键判断：query-map gap 大于 map-side reconstruction gap，继续单独优化 feature reconstruction cosine 的边际收益，不如直接围绕 query-map matcher、flow confidence 和 pose solver 做闭环。`EXPERT_REVIEW_BRIEF_2026-05-07.md`  
现在进一步看，问题甚至更具体：**query-map feature 不只是 gap 大，而是没有形成可泛化的 pose energy。**

---

# 3. 不能转 2D-3D，那应该转什么？

我建议转成：

$$
\boxed{
\text{Pose-conditioned neural energy / denoising refiner}
}
$$

也就是不要再让模型只输出一个 candidate score，而是让模型学习：

$$
F_\theta(I_q, R_{\mathcal{M}}(T), T)
\rightarrow
\left(
E_\theta(T),
\Delta \xi_\theta(T),
\sigma_\theta(T)
\right)
$$

其中：

- $E_\theta(T)$：这个 candidate pose 的 learned energy；
- $\Delta \xi_\theta(T)$：从当前 pose 到 GT pose 的 learned residual；
- $\sigma_\theta(T)$：这个 residual / energy 的可信度。

推理时：

```text
K64 candidates
  -> render map feature/RGB/depth/mask at each T_k
  -> PoseEnergyNet predicts E_k and Δξ_k
  -> optionally apply 1-3 learned updates
  -> select lowest energy / highest confidence
```

这仍然是 learning-based、end-to-end、feature-field-based。  
它不是显式 2D-3D，也不是 PnP。

---

# 4. 为什么 3DGS 新视角扩充训练集可能是关键

我认为：**会有帮助，而且很可能是你现在最应该做的方向。**

原因很简单：你现在的 selector / metric 在 train 上能过拟合，val 上不泛化。这说明当前真实训练 query 覆盖的 pose 分布太稀疏。新视角合成正好可以补这个洞。

这不是拍脑袋。LENS 这类工作明确利用 NeRF 新视角合成扩充相机位姿回归训练集，并指出相机位姿回归很受训练数据分布偏置影响；他们用合成且几何一致的新视角训练 pose regressor，在 benchmark 上显著提升性能。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v164/moreau22a/moreau22a.pdf))  
NeFeS 这类工作也说明了一个相关方向：不是只做 RGB 回归，而是渲染 dense novel-view feature，并与 query feature 做 feature-metric refinement。([arXiv](https://arxiv.org/html/2303.10087v2))  
而 3DGS 的优势是渲染足够快、可见性处理好，官方论文/页面也强调其高质量、实时新视角合成能力。([Inria Repository](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/))

所以你完全可以把 3DGS 的新视角能力变成论文优势：

$$
\boxed{
\text{用 3DGS / DCFF 生成 pose-dense synthetic queries，训练 pose-conditioned feature energy。}
}
$$

这比继续在几十/几百张真实 query 上训 selector合理得多。

---

# 5. 但不能只是“渲染更多图像然后训 selector”

这里要小心。3DGS augmentation 有两种做法，一种有用，一种很可能继续失败。

## 失败版

```text
render RGB_syn
render map feature at same pose
train selector / feature score
```

如果 query 和 map 都来自同一个 renderer，模型会学到 renderer artifact。到了真实 query，它还是不泛化。

也就是说，不能做成：

$$
\text{rendered query} \leftrightarrow \text{rendered candidate}
$$

的简单配对。

## 有用版

应该做成：

```text
render synthetic query RGB
  -> 加强真实相机风格扰动
  -> query student 提 feature

map branch
  -> 渲染 DCFF feature / depth / mask

train:
  query feature 与 map feature 对齐
  pose energy 对候选扰动有正确排序
  learned residual 能把 T 拉回 GT
```

也就是：

$$
I^{syn}_{q}
=
\operatorname{Aug}
\left(
R_{\text{RGB}}(\mathcal{G}, T^*)
\right)
$$

$$
Q^f_{syn} = E_q(I^{syn}_{q})
$$

$$
M^f_k = R_f(\mathcal{M}, T_k)
$$

$$
F_\theta(Q^f_{syn}, M^f_k, T_k)
\rightarrow
E_k,\Delta \xi_k
$$

监督来自：

$$
T^*, T_k
$$

而不是来自显式 2D-3D。

这才符合你的 learning-based / 定位引导重建目标。

---

# 6. 我建议的新主线：NVS-PoseEnergy

我会把下一阶段改成一个非常明确的主线：

$$
\boxed{
\text{3DGS Novel-View Supervised Pose Energy Learning}
}
$$

简称可以叫：

```text
NVS-PoseEnergy
```

或者：

```text
RenderAug Pose-Energy CPR
```

整体结构：

```text
3DGS/DCFF map
   |
   +-- render synthetic query RGB at T*
   |       |
   |    strong query-style augmentation
   |       |
   |    query encoder / projector
   |
   +-- render map feature at candidate T_k
           |
        PoseEnergyNet(Q, M(T_k), aux)
           |
        energy E_k + residual Δξ_k + confidence
```

推理时仍然是：

```text
real query
  -> query feature
  -> K64 candidates
  -> render map feature for each candidate
  -> PoseEnergyNet
  -> select/update pose
```

没有显式 2D-3D，没有 PnP。

---

# 7. 具体训练样本怎么生成

对每个 synthetic GT pose $T^*$，采样一个 candidate bank：

$$
T_k = \exp(\xi_k)T^*
$$

其中 $\xi_k$ 覆盖：

```text
small:  2cm / 0.5°, 5cm / 1°, 10cm / 2°
medium: 25cm / 5°
hard:   50cm / 10°
large:  1m / 20°
```

每个 $T^*$ 采 K64 或 K128 candidates。

对每个 candidate 有真实 pose cost：

$$
c_k =
\|t_k - t^*\|
+
\lambda_R d_R(R_k, R^*)
$$

然后训练 target distribution：

$$
p_k =
\frac{
\exp(-c_k/\tau)
}{
\sum_j \exp(-c_j/\tau)
}
$$

模型输出：

$$
s_k = -E_\theta(Q, M(T_k))
$$

loss：

$$
L_{\text{energy}}
=
-\sum_k p_k \log \operatorname{softmax}(s)_k
$$

同时让网络预测 residual：

$$
\Delta \xi^*_k = \log(T^*T_k^{-1})
$$

$$
L_{\text{residual}}
=
\operatorname{Huber}
\left(
\Delta \xi_\theta(Q,M(T_k)),
\Delta \xi^*_k
\right)
$$

还可以加一个 improvement loss：

$$
T'_k = \exp(\alpha \Delta \xi_\theta)T_k
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

这三个 loss 比当前 selector loss 强很多，因为它们不仅告诉模型“谁最好”，还告诉它“当前 pose 应该往哪里修”。

---

# 8. 这和现在 failed selector 的区别

现在 failed selector 学的是：

$$
\{x_k\}_{k=1}^{64} \rightarrow \arg\min c_k
$$

而且 $x_k$ 只是 local corr/RGB/prior 的统计。

新的 PoseEnergyNet 学的是：

$$
(Q, M(T_k), T_k) \rightarrow E_k, \Delta \xi_k
$$

也就是说，它学习的是 **pose-conditioned residual field**：

$$
T \mapsto \log(T^*T^{-1})
$$

这比 topK selector dense 得多，也更符合 CPR。

---

# 9. 关键：synthetic query 必须做强 domain randomization

如果只用干净 3DGS render，模型会在真实 query 上崩。你需要把 synthetic query 做成“困难版真实 query”。

建议 augmentation：

```text
photometric:
  brightness / contrast / gamma / color temperature
  Gaussian noise / shot noise
  blur / motion blur / JPEG
  vignetting / exposure shift

geometric:
  resize-crop
  small focal jitter
  principal point jitter
  rolling-shutter-like row shift, optional

occlusion:
  random rectangle
  random dynamic-object mask
  alpha dropout
  patch erasing

feature-level:
  channel dropout
  spatial dropout
  random high-pass / low-pass mixing
```

注意：map render branch 不加同样的增强。  
query branch 加强扰动，map branch保持稳定。这样模型学的是：

$$
\text{realistic query style} \rightarrow \text{stable map feature}
$$

而不是 renderer-to-renderer shortcut。

---

# 10. 真实 query 不能丢，要和 synthetic 混训

synthetic 解决 coverage，real 解决 domain gap。

训练 batch 建议：

```text
50% synthetic novel views
50% real training queries
```

real query 上没有那么密的 candidate residual supervision也没关系，至少有：

$$
Q^f_{\text{real}}
\leftrightarrow
M^f(T_{\text{gt}})
$$

可以做 anchor：

$$
L_{\text{real-align}}
=
1 -
\cos
\left(
Q^f_{\text{real}},
R_f(\mathcal{M}, T_{\text{gt}})
\right)
$$

并且采 real 的小扰动 candidates：

$$
T_k = \exp(\xi_k)T_{\text{gt}}
$$

同样可以做 energy/ranking target，因为 pose GT 是有的。只是数量少，但能校准 real domain。

总 loss：

$$
L =
L_{\text{energy}}
+
\lambda_r L_{\text{residual}}
+
\lambda_i L_{\text{improve}}
+
\lambda_a L_{\text{align}}
+
\lambda_d L_{\text{query-drift}}
+
\lambda_t L_{\text{teacher-anchor}}
$$

其中 query drift 很重要，因为你之前看到直接解冻 query fine 后：

$$
map\_q\_f: 0.80 \rightarrow 0.54
$$

所以必须强 anchor。

---

# 11. 推荐的训练阶段

## Stage A：冻结 map，只训练 query projector + PoseEnergyNet

不要再动 map field。先让 evidence 学出来。

```yaml
train:
  map_field: false
  query_backbone: false
  query_projector: true
  pose_energy_net: true
  residual_head: true
```

loss：

```text
synthetic energy CE
synthetic residual Huber
synthetic improve loss
real GT align
real small-perturb energy CE
query drift anchor
RADIO teacher anchor
```

目标：

```text
synthetic-val K64 pred mean < 120mm
real-val 0.25m/5° K64 pred mean 从 250mm 降到 150mm 以下
Spearman 从 0.05 提到 0.25+
```

先别奢望一步到 80mm oracle。

---

## Stage B：加入 rerank adapter，但仍冻结 base map

如果 Stage A 有效，再开一个小 rerank adapter：

$$
F^r = F^f_0 + \alpha A_r(F^f_0)
$$

candidate/energy loss 只更新：

```text
rerank_adapter
query_projector
pose_energy_net
```

不更新：

```text
hash MLP
latent
geometry
base fine decoder
```

防止再次把中 basin 拉坏。

---

## Stage C：小学习率 localization-guided reconstruction

最后才做定位引导重建：

```yaml
map_fine_decoder_lr: 1e-5
rerank_adapter_lr: 5e-5
query_projector_lr: 5e-5
pose_energy_lr: 1e-4
```

map loss 必须有 replay：

```text
teacher anchor
real GT align
synthetic GT align
energy loss
drift from safe-map checkpoint
```

否则又会出现 candidate-gradient 拉坏 feature field。

---

# 12. 代码上怎么落地

我建议新增三个东西，而不是继续改 selector cache。

## 12.1 `export_nvs_pose_energy_cache.py`

功能：

```text
sample synthetic poses T*
render RGB_syn
apply augmentation
sample candidate bank {T_k}
render map features M(T_k)
save pose costs and residual labels
```

输出：

```python
{
  "rgb_syn": [3,H,W],
  "pose_gt": [4,4],
  "candidate_poses": [K,4,4],
  "candidate_pose_cost": [K],
  "candidate_residual": [K,6],
  "render_features": optional or online render,
  "render_mask": optional,
  "render_depth": optional
}
```

如果磁盘压力大，不缓存 render_features，只缓存 `pose_gt + candidate_poses + rgb_syn`，训练时在线 render candidate features。

---

## 12.2 `PoseEnergyNet`

输入：

```text
query fine feature
candidate rendered fine feature
candidate coarse feature optional
depth/mask optional
pose delta from T0 optional
```

输出：

```text
energy_logit [B,K]
residual_delta [B,K,6]
confidence [B,K]
```

结构可以先简单：

```text
local correlation volume / score map
  -> small CNN
  -> candidate token
  -> set transformer or context encoder over K
  -> energy + residual + confidence heads
```

不要只输出 scalar。必须输出 residual。

---

## 12.3 `train_pose_energy.py`

训练：

```text
synthetic + real mixed batch
online render candidate map features
energy CE
residual Huber
improve loss
align/drift anchors
```

eval：

```text
--fine-select pose_energy
--learned-update-steps 0/1/2/3
```

推理路径：

```text
K64 candidates
  -> energy select
  -> optional learned residual update
  -> re-render
  -> second energy select/update
```

---

# 13. 如何判断 3DGS augmentation 是否真的有效

不要直接看最终论文指标。先做四个 diagnostic。

## 13.1 Synthetic → synthetic

训练 synthetic，验证 synthetic held-out poses。

目标：

```text
K64 pred mean 接近 oracle
Spearman > 0.6
AUC > 0.85
```

如果 synthetic 都做不到，说明 PoseEnergyNet / loss 有问题。

## 13.2 Synthetic → real

只用 synthetic 训练，在 real val 上测试。

目标：

```text
real Spearman 从 0.05 提到 0.15+
pred mean 从 250mm 降到 200mm 左右
```

如果 synthetic 好、real 差，就是 domain gap。

## 13.3 Synthetic + real anchor → real

混合训练后 real val。

目标：

```text
0.25m/5° pred mean < 150mm
map_q_f >= 0.70
```

## 13.4 Synthetic + real + residual update

启用 learned update。

目标：

```text
0.25m/5° < 120mm
0.5m/10° < 80-100mm
```

这时才有希望逼近 K64 oracle。

---

# 14. 这条路线如何服务“定位引导重建”

你原来想做：

$$
L_{\text{retrieval/refine}}
\rightarrow
\text{map feature reconstruction}
$$

现在应该改成：

$$
L_{\text{pose-energy}}
+
L_{\text{residual}}
+
L_{\text{improve}}
\rightarrow
\text{rerank adapter / map feature decoder}
$$

也就是说，不要让 map 接收“谁是 top1”的稀疏梯度，而是让 map 接收：

$$
\text{这个 pose 的 energy 应该更低}
$$

$$
\text{这个 pose 的 residual direction 应该指向 GT}
$$

这才是 localization-guided reconstruction。

从论文上可以写成：

> We reconstruct a 3D feature field not merely to imitate RADIO, but to induce a pose-conditioned neural energy landscape for camera pose refinement.

这比 learned selector 更强。

---

# 15. 这是否偏离“可微检索”？

不偏离，只是暂时把检索换成局部 pose bank。

训练时：

$$
\{T_k\}_{k=1}^{K}
$$

可以来自 random K64 / local cube / future retrieval bank。

loss 是：

$$
\operatorname{softmax}(-E_\theta(Q,T_k))
$$

这本质上仍然是可微检索 / 可微候选选择，只不过 score 不是手工 fine similarity，而是 learned pose-conditioned energy。

未来接全局检索时也一样：

```text
global retrieval gives pose bank
PoseEnergyNet scores / updates pose bank
```

所以这条路线和最早目标兼容。

---

# 16. 我会停止的东西

在这个新主线下，以下都该停：

```text
frozen K64 selector
score-map selector继续调
query gate 小修
RGB L1 rerank
lightweight scene-coordinate head
WLS step size
map-side candidate-gradient
coarse adapter继续训
```

这些最新 no-go 已经足够明确了。继续做只会拖慢。

---

# 17. 我会保留的东西

```text
safe-map + adapter checkpoint
random/jitter K64 eval
uniform K64 oracle
score-map diagnostics
map_q_f tracking
AUC / Spearman diagnostics
query drift guard
```

这些都继续有用。

---

# 18. 投稿目标如何调整

不要说：

```text
我们解决了 learned topK selector。
```

这个现在证据不支持。

可以说：

```text
我们发现直接 RADIO/DCFF similarity 不能形成可泛化 pose evidence，
于是提出 novel-view-supervised pose-energy learning，
用 3DGS 合成 pose-dense training queries，
学习定位引导的 map/query feature reconstruction。
```

这反而是一个更有研究味的故事：

1. RADIO teacher feature 不等于 localization feature。
2. 直接 selector 不够。
3. 需要 pose-conditioned energy。
4. 3DGS 新视角合成提供密集 pose supervision。
5. localization loss 反向塑造 feature field。

---

## 最终建议

你不应该转显式 2D-3D；但也不应该继续调现有 selector。下一步应该转成：

$$
\boxed{
\text{3DGS novel-view augmentation}
+
\text{pose-conditioned neural energy}
+
\text{learned residual refinement}
+
\text{strong query-map anchors}
}
$$

3DGS 新视角扩充训练集**很可能会好一些**，但前提是它不是普通的数据增强，而是用于训练一个 pose-conditioned energy / residual field：

$$
(Q, R_{\mathcal{M}}(T))
\rightarrow
E(T), \Delta \xi(T)
$$

这样你的路线仍然是 end-to-end learning-based，仍然是定位引导重建，而且能绕开当前 “feature similarity 本身不可排序” 的死胡同。

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/ICLPose/radio/docs/superpowers/plans/2026-05-11-cpr-selector-metric-no-go-and-next.md)
- [](https://proceedings.mlr.press/v164/moreau22a/moreau22a.pdf)
- [Neural Refinement for Absolute Pose Regression with Feature Synthesis](https://arxiv.org/html/2303.10087v2)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)