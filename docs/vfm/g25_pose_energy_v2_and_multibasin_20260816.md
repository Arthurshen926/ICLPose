# G25 Pose Handoff V2：双侧 latent child、完整 6D 审计与多 basin 搜索

> **已被 V3 正确性审计取代。** 本文中的 V2 数值早于 raw-distorted
> RADIO→ideal-pinhole 坐标重映射、排他的 typed-mass 分解、固定 unknown
> failure floor，以及统一 left-SE(3) 的 73 点拟合。因此旧 Hessian 只能视为
> 诊断历史，不能用于方法晋级。当前结论见
> `g25_pose_energy_v3_correctness_and_hierarchical_atlas_20260816.md`。

## 本轮裁决

附件指出的三个 P0 均已落实并验证：

1. chart layout 改为联合 `(cell, child)` 概率归一化；
2. 完整 sparse child evidence 与 area-selected render payload 在接口上分离；
3. map 侧不再只输出 dominant child，RADIO feature 只与同一个 matching child
   contributor 耦合。

同时完成了完整 6D 二次模型。结果否定了上一轮过强的局部可观测性解释：单轴
切片全部良好，并不意味着完整 6DoF 邻域是局部极大。

## Chart acquisition P0 对照（530 queries）

所有数值仍属于 development-exposed official-test diagnostic。

### Evidence 截断

| 4×4 joint atlas | Top1 2m/45 | Top32 | Top64 |
|---|---:|---:|---:|
| area-selected global | 19.81% | **69.43%** | **75.47%** |
| full sparse global | 22.64% | 68.11% | 75.28% |

完整 evidence 提高 Top1，但当前 tail/noisy children 会轻微伤害多模态深召回。因此
完整 evidence 应保留给 orientation/energy/ambiguity，不能未经校准直接替代面积集合
的 global acquisition queue。

### Joint layout normalization 与分辨率

| scorer | Top1 2m/45 | Top32 | Top64 |
|---|---:|---:|---:|
| 4×4 joint, layout weight .5 | 24.34% | 65.47% | 72.64% |
| 9×16 joint, layout weight .1 | 23.96% | 68.30% | 74.72% |
| 9×16 joint, tolerance 1 cell | **24.15%** | 68.49% | 74.53% |
| 9×16 dual queue 3:1 | 22.64% | 68.11% | 74.72% |

联合归一化修正了概率语义，9×16 与空间容忍也确实增强了 Top1 orientation/layout
信号，但线性融合和简单双队列仍不能逼近 2m/20° atlas oracle 76.98%。目前 deep
recall 仍由 area-selected global queue 最好。下一版需要 full-token 局部 marginalization，
不能继续只调 layout weight。

## Bidirectional Soft Surface Energy V2

Renderer 对每个 token 返回：

```text
child_rows        [H,W,L]
child_weights     [H,W,L]
child_features    [H,W,L,D]
child_feature_valid
null_weight
```

全 geometry 始终参与遮挡；`scene_child_rows` 只裁剪 feature payload，不裁剪 child
identity mass。能量使用：

```text
overlap_u = q_null*r_null + sum_c q[u,c] r[u,c](T)
id_u      = 2*overlap_u - 1
feat_u    = -1 + sum_c q[u,c] r[u,c](T) valid[u,c] (cos[u,c]+1)
score     = mean((1-alpha)*id_u + alpha*feat_u)
```

因此错误 child 的 RADIO feature 不能替另一个 matching child 提供正证据；未匹配质量
固定停留在 -1 floor。真实 `seq13/frame00001` 的 coarse candidate verification 仍把
1.583m/6.55deg basin 从 rank6 提升到 rank1；`seq3` 的正确 basin从 rank4 仅提升到
rank3，仍未成为Top1。

V2 也暴露了此前被条件归一化隐藏的问题：`seq13` GT 附近 mean child overlap 约
0.058，coupled feature mass 约0.040。当前 objective 大部分仍处于 unmatched floor，
所以不能靠调一个 alpha 解决。

## 完整 6D Hessian 审计

实现了两种规范化李代数邻域设计：

- 73 点对称 central-difference design；
- 28 点 minimal full-rank design，用于当前慢 renderer 的 bounded audit。

坐标按 `0.5m` 和 `5deg` 归一化，报告的是 `H(-score)`。审计过程中修复了
`rotate_camera_local` 原地归一化 axis view、从而污染 Hessian probe coordinates 的
实现 bug，并加入不变性回归。

`seq13/frame00001` 的 28 点 V2 结果：

```text
eigenvalues(H(-score)) =
[-0.034713, 0.007509, 0.026238, 0.044468, 0.093296, 0.174827]
```

矩阵不是正定。虽然六个单轴 `H(-score)` 对角切片均为正，完整平移—旋转交叉方向
存在明确鞍点。因此当前 energy 尚不具备从 2m/20° basin 稳定连续收敛的理论条件。
按 oracle ladder 的停机规则，现阶段不应继续扩大 atlas 或直接跑530-query optimizer。

## Multi-basin optimizer

已实现 batched 13-probe SE(3) pattern/trust-region 核心：

- 每个 sweep 一次 evaluator batch 包含所有 basin；
- 每个 basin 独立接受更新或缩小 trust radius；
- 不把弱 basin 静默并入强 basin；
- 不要求 renderer 可微；
- 合成双模态目标可稳定收敛并保留两个 basin。

该 optimizer 暂未接真实全链路。原因不是代码缺口，而是 V2 的完整 Hessian尚有负
方向；此时运行真实 optimizer 只会把 objective 的偏置包装成 pose update。

## Renderer 性能

当前 warm path 的 V2 exact verification：

```text
6 poses render total: 16.133 s
mean:                 2.689 s / pose
score total:          0.104 s
pose batching:        false
```

瓶颈几乎全部在 renderer，而不是 soft energy。Top64 逐 pose exact 仍需约 172 秒/query，
不能用于完整实验。P1 必须是 geometry/code/owner 常驻GPU、pose batch rasterization、
device-side child segmented reduction；简单在 Python 外层循环加 batch 名义没有意义。

## 下一步严格顺序

1. 实现 resident batched Top-L renderer，并保持 full geometry occlusion；
2. 用 GT-visible/full-map 条件在更多 query 上运行完整 6D Hessian；
3. 提高 coupled mass：full-token spatial marginalization、pose-sensitive low-D RADIO
   readout，以及后续 normal/relative-depth/boundary；
4. 只有当 `H(-score)` 在代表性 query 上稳定正定，才接真实 multi-basin trust-region；
5. 先测 2m/20° oracle initializations 的 0.5m/5° capture rate，再回到实际 chart queue。

本轮没有使用 ALIKE、PnP、hard correspondence、新 SfM、新 3DGS 或 query GT ranking。
