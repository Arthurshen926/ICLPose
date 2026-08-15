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
- 530/530 query 的最大损失项都是 C4：正确的正证据 child 已存在，却被
  scene Top64 移除；C4 平均占可见质量 74.94%；
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

10% 下的 GT-only oracle allocation ceiling 为 94.75% exact，说明当前
无 GT 选择器已解决大部分固定 TopK 损失，但仍有约 8.05 个百分点的
posterior 排序/分配差距。90% candidate-posterior 自适应停止版本平均只用
5.84% 面积，达到 69.64% exact、91.10% tolerant、0.764m P90；它是当前
默认成本点，固定 10% 是高召回点。

一米 child 只作为精确 primitive partition。返回后按共享 voxel face、
相同 unsigned dominant-normal axis 和 30 度法向阈值跨 5m parent seam 合并，
不改变 primitive union 或上述召回。10% 点从平均 1,180.8 child 压缩到
294.1 个 connected variable-scale supports（4.02x）；自适应点为 878.4
child / 283.9 supports。这里的 component 是召回区域 carrier，不是伪
landmark，也不代表位姿已经可解。

把 NPZ 解压排除后，在一张真实冻结后验（775 个自适应 child）上重复
50 次，预计算 map-area ledger 后选择器 median/P95 为 24.85/58.24ms，
连通合并为 11.89/12.16ms，合计 median 36.64ms。当前约 3.17s/query 的
主耗时仍是 RADIO mapper/parent-child posterior，不是这次新增的预算器。

完整证据：

- P0 归因：`output/g25_pure_retrieval/full530_geometry_complete_multimode_v1/child_loss_decomposition.json`
- 1/2/5/10% 曲线：`output/g25_pure_retrieval/full530_geometry_complete_multimode_budget_area{01,02,05,10}_v1/evaluation.json`
- 自适应默认：`output/g25_pure_retrieval/full530_geometry_complete_multimode_budget_adaptive90_area10_v1/evaluation.json`

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
2. 校准/锐化 child posterior，目标是在同 5--6% 面积下缩小当前选择器与 GT-only allocation 的差距；不先增加 descriptor 容量；
3. 构建 coverage-complete、surface-connected、overlapping contextual parents，降低当前 parent 21.33% 的地图面积负担并消除 5m hard-voxel seam；
4. 将 connected support 作为后续高容忍度 basin 的 set-valued 输入，但在 fine-support 0.5m recall 稳定越过 95% 前保持 pose handoff 关闭；
5. out-of-map/null 必须增加真实 OOD queries 后才评 gate；当前 in-map-only ECE 只是 false-null diagnostic。

只有 pure retrieval 在更小区域预算下稳定后，才进入 correspondence-free、高容忍度 pose basin 估计；ALIKE/PnP/SfM 不回到主线。
