# G25 纯 RADIO 物理区域召回：覆盖完整性修复与 530-query 验证

## 本阶段问题边界

本实验只回答“query 的 RADIO 全 token 证据能否在冻结 2DGS 物理地图中召回正确区域”。在线路径不读取 query pose/GT，不使用 ALIKE、PnP、SfM point/track、mapping RGB 或参考图像检索，也不生成或渲染候选位姿。GT contributor 只由独立 evaluator 在 pose-free 结果落盘后打开。

方法链为：

```text
query RADIO-final 36x64
-> surface mapper（一次）
-> 每 token 的 parent 匿名多模式后验 + 显式 out-of-map/tail
-> 固定 4x4 空间块的去重复 scene evidence
-> 全局 parent region Top64
-> parent 条件下的 child posterior
-> posterior mass / 2DGS surface-area budgeted set selection
-> 6-connected、法向一致的 variable-scale physical supports
```

parent 是全局物理区域；child 是 parent 内的细表面支持，不是 landmark 或 3D correspondence。

## 修复的结构性错误

旧 physical membership 只让 76,407/509,572 primitives 属于可召回 parent，导致 official 530 queries 的平均可表示可见质量上限仅 19.84%。其余 primitives 虽参与 2DGS 遮挡，却无法成为召回目标。

新增 geometry-complete control hierarchy：

- 冻结原 509,572 primitives，不训练/重建 3DGS；
- 每个 primitive 恰属于一个 child 和一个 parent；
- child：1m world voxel × canonical unsigned dominant-normal axis；
- parent：5m world voxel context cell；
- 法向双侧化，构建器不接受 pose/image/route/RADIO/GT 输入；
- canonical RADIO field 使用 raw SIMPLE_RADIAL -> ideal contributor 的明确坐标合同；
- 用独立 seq11 的 23 帧拟合 in-map validity calibration。

该层级是“覆盖完整性与召回能力”的最小控制组，不是最终 contextual physical chart。5m hard voxel 会包含多种外观/表面，仍需更合理的连通、重叠与多模态 chart 表征。

## 四组完整对照

| 530-query 指标 | 旧 membership + 旧坐标 | 旧 membership + 坐标修正 | 全几何 + 单均值 | 全几何 + 匿名多模式 |
|---|---:|---:|---:|---:|
| parent 可表示可见质量上限 | 19.84% | 19.84% | **100.00%** | **100.00%** |
| parent token conditional R@32 | 95.58% | 95.59% | 95.15% | **97.19%** |
| parent token absolute mass@32 | 18.76% | 18.76% | 95.13% | **97.17%** |
| parent Top64 精确可见表面召回 | 16.67% | 16.70% | 93.56% | **95.05%** |
| parent Top64 0.5m 容忍表面召回 | 37.59% | 37.59% | 95.02% | **95.83%** |
| parent Top64 表面距离 P90 | 4.361m | 4.350m | 0.399m | **0.343m** |
| parent Top64 地图表面面积比 | - | - | **19.34%** | 21.33% |
| child token conditional R@64 | 87.25% | 87.23% | 84.91% | **87.98%** |
| child Top64 精确可见表面召回 | 3.83% | 3.87% | 18.26% | **19.88%** |
| child Top64 0.5m 容忍表面召回 | 14.91% | 14.93% | 34.69% | **37.73%** |
| retrieved-set geometric non-degeneracy@64 | 99.62% | 99.62% | **100.00%** | **100.00%** |
| 单 query 平均 wall time | 1.92s | 2.02s | 2.85s | 3.17s |

坐标修正是必须的正确性修复，但在旧层级上几乎不改变性能；决定性瓶颈是物理 membership coverage。覆盖修复后，parent 内部的多视角/遮挡模式确实会被单均值压坍：最多 4 个匿名模式将 R@32 提高 2.04 个百分点，并将完全漏召回从 12/530 降到 2/530。

## 防止“大区域刷高召回”的选择性审计

全几何单均值 parent Top64 平均覆盖 3,482.67m²，占全地图 18,007.26m² 的 19.34%，召回 93.56% query-visible mass，exact-recall/area-fraction enrichment=4.84x。匿名多模式把精确召回提高到 95.05%，但面积也增到 21.33%，enrichment 降到 4.46x。

固定“最多全地图 20% 表面面积”后，单均值实际使用 18.20% 面积，exact/0.5m recall=92.76%/94.47%；匿名多模式实际使用 18.95%，exact/0.5m=93.81%/94.84%。这证明多模式有小而真实的 recall-area 改进，但固定 Top64 下的较大收益部分来自选中了更大的区域，不能声称无代价全面胜出。

另一个“parent evidence/表面面积”的 fractional-knapsack 对照在 seq11 同 20% 面积下将 exact/0.5m 从 94.01%/94.33% 小幅提高到 94.26%/94.77%，p90 从 0.569m 降到 0.210m，但候选 parent 数从 99.4 暴涨到 352.9。它因每区域固定开销过大而不晋级，只作 recall-area 诊断控制。

child Top64 只占地图表面积 1.21%，exact enrichment=15.06x，但其绝对召回不足。新增同预算层级控制“Top16 parent × 每 parent Top4 child”只占 0.93% 地图表面，0.5m recall=35.01%；固定平均配额没有优于全局 child Top64，说明 child 预算必须由 parent posterior、局部不确定性和表面质量自适应分配。

## P0 child loss decomposition 与预算选择修复

在不改变冻结检索结果的前提下，独立 evaluator 对 530 张 query 做了
C0--C7 归因。结果排除了“先训练更强 child descriptor”这一错误方向：

- 当前 Top64 child exact recall：19.88%；
- 全 token child candidate union exact ceiling：98.61%；
- 当前 parent 内 candidate union ceiling：94.53%；
- oracle parent + 当前 candidate ceiling：98.61%；
- 530/530 query 的最大损失项都是 C4：正确的正证据 child 已存在，却在
  下游排序、固定预算与 set construction 中丢失；C4 平均占可见质量
  74.94%。C4 不是“纯预算损失”，不能把排序和集合构造残差错误归因给预算；
- C0 无 canonical feature 仅 0.06%，C1 parent 未选 4.95%，C2 parent 内
  child 未进入 token candidate 仅 0.53%。

因此新选择器直接优化检索目标：对 2304 个 token 的 child posterior
质量求和，以 child 所代表的唯一 2DGS ellipse area 为成本，在已召回
parent 内做确定性 density-knapsack，并用 primitive IoU 去除重复支持。
它不读 pose/GT，不使用图像对应，也不把 child 当 landmark。实际 530
query 面积曲线为：

| 最大地图表面预算 | exact visible mass | 0.5m tolerant | distance P90 | 平均 child |
|---:|---:|---:|---:|---:|
| 1% | 19.98% | 66.31% | 1.501m | 429.0 |
| 2% | 34.60% | 77.98% | 1.146m | 565.7 |
| 5% | 64.61% | 89.92% | 0.790m | 832.9 |
| 10%（实际 9.98%） | **86.70%** | **94.34%** | **0.497m** | 1,180.8 |

旧的 94.75% GT-only ceiling 使用了更大的 full-map current-parent child
候选空间，不能与当前选择器作严格差值。新增的 matched-candidate oracle
严格使用与无 GT 选择器相同的 child universe、相同 primitive-IoU NMS 和
相同面积账本：1/2/5/10% 面积下 exact ceiling 分别为
42.44/60.79/87.35/**94.27%**，其无面积上限 ceiling 为 94.53%。因此
10% 点真正的同空间排序/分配差距是 **7.57 个百分点**，而不是此前混合
候选空间得到的 8.05 个百分点。90% candidate-evidence 自适应停止版本平均只用
5.84% 面积，达到 69.64% exact、91.10% tolerant、0.764m P90；它是当前
默认成本点，固定 10% 是高召回点。

## child 概率语义与冻结选择器消融

代码审计和逐 query 数值审计确认，检索产物里的 child 值已经是截断的
联合质量

```text
P(parent | token) * P(child | parent, token)
```

而不是裸条件概率 `P(child | parent, token)`。每个 token-parent 下保留 child
的质量之和不超过该 parent 质量；未保留的 tail 明确保留为未分配质量。
因此 artifact 现在显式写
`truncated_joint_parent_child_probability_per_radio_token_v1`。跨 token 求和只叫
`summed_token_evidence_mass`，它不是 calibrated credible mass，也不能解释为
“90% 后验可信区间”。旧参数名保留兼容别名，新主名称是
`target_eligible_evidence_fraction`。

附件建议的 S0--S4 被转成同一 530-query、同一 10% 物理面积预算的冻结消融；
所有区间都是按 sequence 内连续 16 帧 block bootstrap、2,000 次重采样：

| 选择器 | exact | 0.5m tolerant | P90 | 平均 component | 结论 |
|---|---:|---:|---:|---:|---|
| S0 joint token sum | **86.70%** `[84.05,89.36]` | 94.34% `[92.60,95.89]` | 0.497m | 294.1 | 新基线 |
| S2 4x4 block capped sum | 86.70% | 94.34% | 0.497m | 294.1 | 几乎从不饱和，无实际作用 |
| S2 4x4 block max | 86.43% | 94.42% | 0.496m | 301.5 | exact 降、碎片增，拒绝晋级 |
| S4 hard component cap=220 | 79.22% | 91.92% | 0.717m | 217.6 | 删除真实多解支持，明确拒绝 |

S1 没有产生新数值分支，因为当前实现本来就是联合 parent-child evidence；
S3 的“单 token child marginal cap”在当前互斥 child partition 下与 S0 数学等价：
同一 token 的 child 联合质量和已经不超过 1。这个结论避免了为了名称变化重复跑
一套完全相同的实验。S4 的失败尤其重要：component 数不是可以免费压缩的冗余，
硬截断会系统性删除真实歧义和长表面。后续 multiscale support 必须保持原
primitive union，以重叠 carrier 表达同一物理证据，而不能靠删除 component 达标。

一米 child 只作为精确 primitive partition。返回后按共享 voxel face、
相同 unsigned dominant-normal axis 和 30 度法向阈值跨 5m parent seam 合并，
不改变 primitive union 或上述召回。10% 点从平均 1,180.8 child 压缩到
294.1 个 connected variable-scale supports（4.02x）；自适应点为 878.4
child / 283.9 supports。这里的 component 是召回区域 carrier，不是伪
landmark，也不代表位姿已经可解。

把 NPZ 解压排除后，在一张真实冻结后验（775 个自适应 child）上重复
50 次，预计算 map-area ledger 后选择器 median/P95 为 24.85/58.24ms，
连通合并为 11.89/12.16ms，合计 median 36.64ms。当前约 3.17s/query 的
主耗时仍是 parent-child posterior，不是这次新增的预算器。进一步把 child
展开从 `2304 x all_children` 的 dense float64 矩阵改为只聚合真实出现的
token-child triples；真实 query 的 token/scene child rows 和 scores 逐位不变。
单张冷启动 child 阶段从 3.236s 降到 2.312s（-28.6%），端到端从 8.060s
降到 7.053s（-12.5%）。随后 profile 发现剩余约一半 child 时间只是在对
全部 token-child triples 做全局 lexsort。改成逐 token kth-threshold、只排序
最终 Top64，并对阈值并列按最小 child row 补齐后，五张连续 query 的暖机
child 中位数进一步从 1.891s 降到 **0.759s（-59.9%）**；端到端稳定在
1.69--1.76s。新旧 5/5 真实查询的完整 retrieval content hash、token child
rows/probabilities 全部逐位一致。当前暖机阶段分解约为 layout 0.376s、parent
0.191s、child 0.759s、集合选择 0.301s；下一性能项才是批量化 parent-group
小矩阵，而不是 mapper 或预算器。

## sequence-disjoint 轻量 reranker 验证

为了验证 matched-candidate 7.57pp gap 是否可由低容量校准缩小，新增严格
leave-one-sequence-out 诊断。每张 query 的特征只包含联合 child evidence、
支持 token 数、最大 token 质量、parent 内 evidence 比例、child 面积/primitive
数/extent；不使用绝对世界位置、route、pose、RGB 或 GT。每次留出 seq13、
seq3 或 seq5，模型只读取另两条 sequence 的 contributor 标签。530 张历史上
已被开发打开，所以该结果只称 sequence-crossfit development diagnostic，
不是 untouched test 或 deployment calibration。

| 10% 面积方法 | exact | 0.5m | P90 | child | component |
|---|---:|---:|---:|---:|---:|
| S0 evidence coverage | 86.70% | **94.34%** | **0.4970m** | 1180.8 | 294.1 |
| crossfit exact-density reranker | **87.79%** | 92.68% | 0.6480m | 854.1 | 171.4 |
| equal-area exact+coverage queues | 87.68% | 94.24% | 0.4977m | 1156.0 | 291.9 |

相对 S0 的配对连续 16-frame block bootstrap：exact-only reranker 为
`+1.08pp [ +0.56,+1.62 ]`，但 0.5m 为
`-1.66pp [ -2.11,-1.29 ]`；不晋级。双队列把 exact 增益保留为
`+0.98pp [ +0.71,+1.24 ]`，并把 0.5m 损失压到
`-0.100pp [ -0.166,-0.037 ]`，三个 held sequence 的 exact 均提高；但容忍
召回仍是可分辨的小幅下降，因此也不替代 S0，只保留为 Pareto 候选。

这个结果定位了下一理论缺口：pointwise child 身份排序能提高 exact，却会把
面积集中到更少、更大的 component；简单队列融合只能近似恢复 halo。要同时
提高 exact 与 tolerance，训练目标和集合构造必须显式建模选中 primitive union
到可见表面的距离覆盖/边际收益，而不是再调 pointwise evidence 权重。

完整证据：

- P0 归因与 matched-candidate oracle：`output/g25_pure_retrieval/full530_geometry_complete_multimode_v1/child_loss_decomposition_v2.json`
- 1/2/5/10% 曲线：`output/g25_pure_retrieval/full530_geometry_complete_multimode_budget_area{01,02,05,10}_v1/evaluation.json`
- 自适应默认：`output/g25_pure_retrieval/full530_geometry_complete_multimode_budget_adaptive90_area10_v1/evaluation.json`
- S0/S2/S4 冻结消融汇总：`output/g25_pure_retrieval/full530_fine_support_ablation_v2.json`
- 暖机分阶段 profile：`output/g25_pure_retrieval/profile5_sparse_child_warm_v2/retrieval.json`
- sequence-crossfit reranker：`output/g25_pure_retrieval/full530_crossfit_child_reranker_area10_v1/evaluation.json`
- exact+coverage 双队列及配对区间：`output/g25_pure_retrieval/full530_crossfit_two_queue_area10_v1/comparison.json`
- 逐 token TopK 优化 profile：`output/g25_pure_retrieval/profile5_tokenwise_topk_v3/retrieval.json`

## 第一性原理结论

1. **正确全局区域召回已被实验证明可行。** 在不生成 pose、不做对应、不用参考图像的情况下，RADIO + 冻结 2DGS 中间表征的 parent Top64 达到 95.02% 的 0.5m sampled-surface recall。
2. **旧方法的首要失败不是 RADIO 本身，而是地图把大多数物理表面排除在可召回表示之外。**
3. **单一 parent 平均 descriptor 是已验证的瓶颈。** 匿名多模式在不把 mode multiplicity 当先验的前提下，将 parent conditional R@32 从 95.15% 提高到 97.19%。
4. **child 不能作为全局统一 TopK pseudo-landmark。** 新的面积约束 posterior-mass 集合选择将 0.5m recall 从 37.73% 提高到 94.34%，证明应返回 set-valued physical support；固定 child-ID@64 不再是主要方法指标。
5. **当前结果是区域召回，不是 pose/localization success。** 表面真值是坐标修正后的 sampled 2DGS contributor mass，不是真实世界 dense geometry truth。
6. **官方 test 历史上已被多轮开发诊断打开。** 本结果可作为完整 530-query 对照，但不能称为 untouched test。

## 下一步（保持同一主线）

优先级只围绕纯召回：

1. 保留 multiplicity-normalized anonymous multimode，并固定报告 1/2/5/10% 面积曲线、返回 connected support 数和 wall time；
2. 保留 sequence-crossfit identity reranker 作为 exact queue，但默认仍使用 S0；
   下一模型训练目标改为固定 0.5m 距离覆盖的边际收益，与 exact identity 分开
   报告，只有 exact 和 tolerance 都不降才晋级；
3. 构建 coverage-complete、surface-connected、overlapping multiscale supports，
   保持 selected primitive union 不变，再比较 support 数、面积、exact/tolerant
   recall；硬 component cap 已被全量实验否决；
4. 将 connected support 作为后续高容忍度 basin 的 set-valued 输入，但在 fine-support 0.5m recall 稳定越过 95% 前保持 pose handoff 关闭；
5. 逐 token threshold 已把 child 从 1.89s 降到 0.76s；下一步按 child-count
   分桶批量执行 parent-conditioned 小矩阵，同时继续要求完整 content hash 逐位一致；
6. out-of-map/null 必须增加真实 OOD queries 后才评 gate；当前 in-map-only ECE 只是 false-null diagnostic。

只有 pure retrieval 在更小区域预算下稳定后，才进入 correspondence-free、高容忍度 pose basin 估计；ALIKE/PnP/SfM 不回到主线。
