# G25 pose energy V3：正确性、证据归因与层级召回

## 结论

本轮不引入 ALIKE、PnP、SfM 或新的 Gaussian 训练。地图仍是已有 3DGS，查询仍由 RADIO 表征；目标是修复从 RADIO/3DGS 区域召回到连续位姿能量之间的理论和实现断点。

当前结论分为三层：

1. child atlas 已能作为高容忍度、多 basin 的前端，但位置和朝向必须分层排序；
2. GT 位姿处存在充足的在线 3DGS 可见质量，主要瓶颈是 query evidence 到 child identity 的匹配，而不是地图支撑缺失；
3. 原 V2 能量的坐标、missingness、局部二次拟合合同均不够严格，旧 Hessian 数值不可继续作为晋级证据。

## P0 正确性修复

### 统一坐标合同

3DGS 在线渲染在 4 倍 supersampled ideal-pinhole canvas 上执行。每个 raw-distorted RADIO sample 通过 SIMPLE_RADIAL 的逆向采样合同映射到 ideal canvas，再以固定 4×4 raw sample 聚合到 36×64 token。不能再把 ideal 1024×576 的整齐 16×16 块直接当作 raw-distorted RADIO token。

稀疏 hit remap 与 dense contributor 坐标实现已做逐项相等测试。

### 排他的 typed mass

每个 token 的 identity 质量分解为：

`Top-L child + child tail + field missing + background = 1`。

payload missing 只描述 feature 是否可用，不再与 identity missing 混为一谈。query/render 两侧 unknown 不产生正匹配；证据消失只能留在固定失败 floor，不能通过 null-null 配对提高分数。

### 先聚合 child，再截断 Top-L

所有正 primitive contribution 先按 child 聚合，之后才截断 Top-L。primitive 级阈值不能先删除属于同一 child 的小贡献。

### Payload 与 token-order 不变性

更换 canonical payload 只允许改变 feature availability，不允许改变 child identity、质量或 typed residual。token 顺序必须精确等于 row-major `(x,y)`；形状相同但排列不同会被拒绝。

### 局部 SE(3) 二次审计

全部 6D probe 统一使用左乘 retraction：

`T(xi) = Exp(xi^) T0`。

完整对称二次模型使用 73 点拟合，报告最小特征值、特征向量、拟合 RMSE/最大残差，并沿最小特征向量直接探测 `±0.5, ±1, ±2`。28 点恰定模型仅保留为快速诊断，不能作为最终曲率证据。

## Matched-evidence 梯度

在 `seq13/frame00001.png` 上得到：

| 层级 | 均值 |
|---|---:|
| query retrieval retained child mass | 0.7548 |
| contributor Top-4 retained child mass | 0.4448 |
| GT-pose online render retained child mass | 0.8411 |
| query retrieval × contributor same-child overlap | 0.03095 |
| query retrieval × GT render same-child overlap | 0.05757 |
| contributor × GT render unconditional overlap | 0.60010 |
| contributor × GT render observed-conditional overlap | 0.91657 |

解释：在线 full-compositing renderer 与 contributor cache 在已经观察到的质量上身份基本一致；绝对差异主要来自旧 cache 只保存 Top-4 primitive。GT render 有足够 child 质量，但 query evidence 只与其重叠约 5.8%，所以主瓶颈是 query-to-child attribution 和空间容忍，不是 3DGS 缺少正确表面。

mapping-pose round-trip 当前仍只能作为诊断：online full compositing 与旧 Top-4 contributor cache 的 unconditional identity overlap 为 0.6411、observed-conditional 为 0.9253。由于两侧截断语义不同，不允许把它伪称 exact round-trip authority。

## 位置—朝向分层 atlas

旧全局排序用一个分数同时回答“在哪里”和“朝哪看”，会让同位置的方向重复挤掉其他空间 basin。新排序分三步：

1. 全局 child evidence 选择空间位置 seed；
2. 仅在每个 2m 空间邻域内用 layout 选择方向；
3. round-robin 分配位置配额，每个位置默认最多两个、相差至少 10° 的方向。

530-query 标准测试结果：

| 指标 | 全局排序 | 分层排序 | 变化 |
|---|---:|---:|---:|
| 2m/45° Top32 | 69.43% | 74.91% | +5.48 pp |
| 2m/45° Top64 | 75.47% | 82.45% | +6.98 pp |
| 2m/20° Top64 | 71.51% | 72.45% | +0.94 pp |

Top1 略降，符合它作为多 basin 高召回前端而不是单假设分类器的定位。该改动证明位置和朝向分层有效，但 82.45% 仍未达到最终系统所需的高覆盖，不能冻结为最终召回器。

需要区分“排序失败”和“atlas 本身无覆盖”：当前 atlas oracle 为 2m/45° 90.19%、2m/20° 76.98%。因此分层 Top64 对可覆盖 query 的条件保留率分别达到 91.42% 和 94.12%。这说明排序前端已接近可用，剩余绝对 recall 的较大部分来自 atlas 空间/方向支持不足；后续不能只继续调排序分数。

## Energy V3 最小实现

V3 使用固定五 token 空间核：中心权重 1/2，四个 cardinal 邻居各 1/8，边界不重归一化。parent overlap 提供空间容忍，child overlap 提供精细身份，同-child RADIO cosine 只在同一 child 质量上耦合。missing/off-grid 始终留在 -1 floor。

`seq13/frame00001.png` 的 GT 中心单轴快速审计中，parent overlap 为 0.2744、child overlap 为 0.0529，parent 提供了约 5.2 倍更强的可观察支撑；六个单轴的 `H(-score)` 均为正。该结果只证明单轴局部方向恢复，完整交叉 Hessian 结果必须单独报告，不能由单轴外推。

同一帧上的 28 点恰定交叉模型给出最小特征值 -0.01414，且拟合残差近似为零；然而沿该特征向量直接评估时，`±0.5, ±1, ±2` 的分数全部低于 GT 中心（中心 -0.78895，直接 probes 为 -0.8050 至 -0.8905）。因此这个负特征值没有被真实能量验证，是“恰定拟合零残差不等于局部模型正确”的实证反例。

V3 的 73 点过定审计最终为正定：六个 `H(-score)` 特征值为 `[0.00964, 0.05488, 0.05925, 0.07210, 0.21798, 0.24231]`，RMSE 0.01019、最大残差 0.03056。它与直接负方向 probes 一致，推翻了 28 点的假负特征值。V3 中心分数 -0.78895，也明显高于相同输入下 V2 的 -0.89037；提升来自 parent 空间容忍支撑，而不是 null/null 奖励。

V3 在 0.5× 邻域同样正定，特征值为 `[0.00259, 0.01342, 0.01520, 0.02451, 0.11758, 0.13391]`，RMSE 0.00414。不过 GT center 在 73 个 probe 中排第 2：`tz-`（0.125m 的单轴 probe）仅高 0.000234。二次模型预测的联合偏移约 0.141m / 0.148°，直接重渲染后反而比 center 低 0.000022。结论不是“存在可靠的 14cm 系统偏差”，而是深度方向在小邻域非常浅，二次模型的 bias 不足以稳定指导精确位姿；V3 当前适合 basin capture/高容忍度 refinement，不足以单独承担最终厘米级 pose readout。

修正后的 V2 在同一帧、同一尺度上的 73 点结果为正定：`H(-score)` 六个特征值为 `[0.01143, 0.06836, 0.07290, 0.08307, 0.20033, 0.20097]`，拟合 RMSE 0.01115、最大残差 0.03176。这个结果撤销了旧坐标/旧 missingness 合同下的“稳定负曲率”结论，但目前仅是一帧单尺度诊断，不能外推为总体 optimizer readiness。

V2 在 0.5× 邻域（0.25m / 2.5°）的第二组 73 点也保持正定，特征值为 `[0.00307, 0.01873, 0.02032, 0.03037, 0.13143, 0.13875]`，RMSE 降至 0.00567。最弱方向仍接近相机前向平移，但没有在缩小邻域后翻成负曲率。

## 尚未晋级的部分

- 旧 contributor cache 是 Top-4 primitive，不具备与 online full compositing 完全相同的 round-trip 语义；需要同语义 cache 或 live replay authority。
- 当前 correctness-first supersampled renderer 仍是逐 pose 标量路径，尚未实现 resident scene + batched coarse/exact 接口；不能开始大规模 multi-basin refinement。
- 单帧 73 点 V2 审计的实测 wall time 约 12 分钟，即约 10 秒/pose（包含 Python 端渲染、坐标 remap 和聚合开销）。这直接确认 resident/batched renderer 是实际系统瓶颈，不是可推迟的工程美化。
- Energy V3 当前是最小 parent→child 空间模型。信息权重、normal、relative depth、boundary 必须逐项消融，不能一次混入后用总指标掩盖退化。
- 只有完整 73 点、多尺度曲率在代表性 query 上不再出现稳定负方向，multi-basin optimizer 才能从 oracle/debug 进入实际候选链。

## 下一执行门

1. 完成修正后 V2 与 V3 的完整交叉 Hessian；若沿负特征向量存在稳定 score 增益则直接 KILL 对应能量。
2. 实现 scene-resident batched renderer，并以 scalar/batch typed arrays、score、排序逐项等价作为硬门。
3. 在通过曲率的能量上运行真实 Top-K multi-basin capture：报告进入 basin、优化后 basin survival、重复率与失败原因，禁止只报 best case。
4. 继续提高空间 seed 覆盖，优先优化 query-to-parent/child attribution 和完整 token layout；不重新引入局部特征匹配或 PnP。

multi-basin pattern search 的 13 点 trust-region probe 已同步改为与 Hessian 完全相同的 left-SE(3) exponential；不再混用 world translation 和 camera-local rotation。该优化器当前通过合成双 basin 不塌缩测试，但在 resident batch renderer 和代表性 73 点曲率门完成前仍不接真实主链。
