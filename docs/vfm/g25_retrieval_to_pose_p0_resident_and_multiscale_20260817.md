# G25 retrieval-to-pose：P0 语义、resident renderer 与多尺度 proposal

日期：2026-08-17

## 1. 本轮边界

本轮不训练新高斯、不使用 ALIKE/PnP/SfM point correspondence，也不改已经冻结的
纯 RADIO child retrieval。输入仍是已有 clean 2DGS/physical map、canonical RADIO
surface field、纯 retrieval posterior 和 pose-free visibility atlas。

目标是修复 retrieval 到连续 6DoF pose energy 之间的三类问题：

1. renderer/energy 的 typed mass 语义不清；
2. 标量 exact renderer 无法支撑 multi-basin 实验；
3. atlas 与 pattern search 仍会过早丢掉位置 basin 或同步收缩不相关轴。

## 2. P0 typed semantics

`RenderedSoftChildMixture` 的 identity partition 现在严格定义为：

```text
Top-L child + child tail + unassigned geometry + background = 1
```

feature-side 缺失独立分成：

```text
canonical_field_missing_weight
payload_excluded_weight
```

二者不再改变 child identity。alpha > 1 的数值越界不会归一化掩盖；超过固定
`2e-5` 容差直接拒绝。canonical/payload mass 超过其物理上层 mass 也直接拒绝。
所有 contribution 按 token、稳定 primitive ID 和权重的固定顺序归约。

真实 `seq11/frame00001`、509,572 primitive、factor-4 raw SIMPLE_RADIAL remap：

| 检查 | 结果 |
|---|---:|
| typed identity maximum error | `4.7684e-7`（初版）/ `2.3842e-7`（segmented reducer） |
| maximum alpha overflow | `0` |
| payload identity fields | 全部 exact |
| mapping contributor vs online mean identity overlap | `0.82431855` |
| observed-conditional overlap | `0.91122655` |
| mean unassigned geometry | `1.12e-7` |

`coupled_feature_fraction` 也已拆成：

```text
coupled_feature_mass
coupled_feature_agreement
```

因此可以区分“没有 feature 质量”和“有质量但 RADIO cosine 不一致”。

## 3. resident exact renderer

新增 `FrozenSoftSurfaceSceneGPU`。以下内容只物化一次并常驻 GPU：

```text
center / quaternion / scale / opacity
normal / sidedness
stable primitive ID
child owner / parent owner
canonical field row / normalized code / confidence / uncertainty
```

相机 raw-distorted sample 到 ideal supersampled pixel/token 的 warp 按完整 intrinsics
key 缓存。batch 路径直接调用 gsplat 的 batched 2DGS projection/tile intersection，
每 camera 用独立 signed-sidedness mask；不再调用会生成无用 RGB/normal image 的完整
`rasterization_2dgs`。全部 geometry 仍参与遮挡，selected child 只限制 feature payload。

### 3.1 Correctness gate

真实 3-pose audit（mapping pose + 两个固定 SE(3) perturbation）：

| 检查 | 结果 |
|---|---:|
| child rows | exact |
| child feature valid | exact |
| scalar/batch maximum numeric error | `5.9605e-8` |
| batch reverse-permutation maximum error | `5.9605e-8` |
| gate tolerance | `2e-6` |
| equivalence gate | PASS |

### 3.2 Performance gate

同一真实 3-pose audit：

| 项目 | 时间 |
|---|---:|
| scalar total | `13.651 s` |
| resident batch total | `8.363 s` |
| relative speedup | `1.632x` |
| resident GPU static payload bytes | `232,963,228 B` |
| resident host child/feature reduction | `7.075 s` |

因此 correctness 已过，但 production speed gate 明确未过。当前唯一主要瓶颈已经从
重复 geometry transfer/rasterization 收缩到 deterministic child/feature segment
reduction。下一实现必须将以下链条移到 GPU：

```text
(batch, raw token, child) key
→ deterministic segment sum
→ per-token Top-L
→ typed residual
→ selected-child feature/coupled cosine reduction
```

在 GPU reducer 完成前，artifact 必须保持 `promotion_eligible=false`。

## 4. 多尺度共享能量与 overlap ladder

hierarchical energy 现在支持固定 radius 0/1/2/3 token kernel；所有 border 缺失
保持 failure floor，不做候选相关重归一化。新增 query-only reliability：

```text
w_u = child_mass × (0.25 + 0.75 × posterior_concentration)
      × (1 - out_of_map_probability)
```

`w_u` 只由 query retrieval 决定，对所有 pose 固定，因此 evidence disappearing
仍不能提高分数。新增 overlap ladder：

```text
parent overlap
child overlap radius-2
child overlap radius-1
child overlap radius-0
```

这避免继续把 exact same-child overlap 当成唯一训练目标。下一阶段可把 radius-2
用于 coarse basin survival、radius-1 用于 medium refinement、radius-0 只用于 fine
survivor，而不改变 identity/missing 语义。

## 5. atlas selection v2

位置分数改为同一位置邻域 orientation 分数的 density-corrected log-mean-exp；位置
邻接图由 `cKDTree` 构建并按 pose content/radius 缓存。fallback 顺序固定为：

1. 新未覆盖 location；
2. 已有 location 的缺失 orientation quota；
3. 普通全局 fallback。

同时去掉了 fallback 前额外执行一次 full-library greedy NMS 的近二次路径。

固定 530-query acquisition（同一 retrieval、同一 atlas、无重训）：

| 指标 | 原 hierarchical v1 | location-marginal v2 |
|---|---:|---:|
| 2m/45° Top32 | 74.91% | 77.36% |
| 2m/45° Top64 | 82.45% | 83.40% |
| 1m/10° Top64 | 20.00% | 15.09% |

结论不是“v2 全面替换 v1”，而是 v2 更适合 coarse wide-basin acquisition；近邻
精细候选需要 medium/fine 分支或更高 orientation quota。该取舍必须在后续 cascade
中显式报告，不能只展示改善的 2m/45°。

## 6. anisotropic multi-basin pattern search

pattern search 现在：

* 分别维护 `[tx,ty,tz]` 与 `[rx,ry,rz]` 六个半径；
* 只有无增益的轴缩小；
* center score 首次计算后跨 sweep 缓存，每 basin/sweep 从13次降到12次 probe；
* `maximum_basins` 只在第一轮 coarse sweep 后应用；
* 提供 location ID 时强制每个 location 至少保留一个 post-coarse survivor，预算不足
  直接拒绝而不是静默删除。

## 7. 当前判定与下一门

已经 GO：typed semantics、scalar segmented reducer、resident batch numerical
equivalence、location-marginal coarse proposal、anisotropic pattern-search core、query-only
reliability 与 overlap ladder。

仍然 KILL：把 resident renderer 称为 production batched renderer；Top8 exact speed gate
尚未通过，coarse 64-pose 20x gate也尚未执行。下一唯一优先实现是 deterministic GPU
child/feature reducer，然后按同一固定 pose 集运行 scalar/batch/permutation/ranking/speed
联合门。通过后才值得打开固定 diagnostic query 的 coarse→medium→fine basin-capture
实验。
