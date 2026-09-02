# G25 fresh exact16 final gate：零信任复核

日期：2026-08-30  
范围：只读复核现有 frozen artifacts；未修改共享 gate；未运行 GPU。

## 审计结论

当前 exact16 报告不能直接解释成“两个 chart atlas 后端都已经被可靠否决”。更准确的结论是：

1. fresh source/held 隔离、AABB、16-chart 共拓扑、四个 atlas、CPU z-buffer 和 held 指标数值都能从原始数组独立复现；这些部分可信。
2. M1/DAV2 有独立于 seam 的严重 chart 尺度畸变，因此 M1 的 geometry KILL 可信。
3. M2/MoGe3 的非 seam distortion 检查通过；它目前的 geometry KILL 完全依赖一个存在 P0 定义问题的 seam 指标。因此 M2 必须在修正 seam 后重新用现有 artifact 做 CPU 评估，不能现在下最终 KILL。
4. M1/M2 的 absolute held coverage 只是点估计 GO。四个时间块的区间下界都远低于门槛，不能把它写成总体或部署层面的正式 GO。

机器可读审计见：

`/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_exact16_fresh/audit/fresh_exact16_zero_trust_gate_audit_v1.json`

## 已独立复现的内容

### 权威链与 bounds

- authority content：`34b9acecf4fa7472bf39697827783ba900d22216d58901107750a1bdb7792a82`
- fresh exact16 plan content：`cd639963a640de6f4861fff56b74143497b954975276b72f73c4f17f787f7e99`
- physical-safe comparison domain content：`04855ad4107c1dd18d17d2f859cd95bbbda027ae42862a40be90295a12da547a`
- source 为 seq4 的 24 帧，held 为 seq1 的 12 帧；名字交集为 0，物理 root 不同。
- 从 exact face-referenced M0、DAV2 和 MoGe3 原始 initializer 顶点重新计算 bounds，和 artifact 逐位一致：
  - min = `[-1.8316948509317967, -42.49961181916812, -5.479687332669407]`
  - max = `[20.368767948318897, -2.364964841138203, 25.580157569553222]`
  - 最大绝对误差为 0。

held 仍然只是 source-disjoint 的 MASt3R mapping geometry diagnostic，不是 Cambridge 传感器深度真值，也没有使用 query/GT。

### exact topology

正式 gate 确实只使用 stride4：16 个 chart 全部非空，共 3324 个顶点、4062 个三角形。M0、M1 initial/aligned、M2 initial/aligned 的 chart 顺序、vertex offsets、face offsets 和 faces 完全一致；四个 atlas 的 UV 也完全一致。stride8 没有进入正式 gate。

### z-buffer、分母与 normal

我没有调用报告中的聚合结果，而是从五个 surface NPZ 和 held ray NPZ 独立实现 CPU raster/recount。三个 arm 的逐视图整数 paired/good/normal20/joint20 计数与报告完全一致，浮点差为 0（最后舍入精度内）。没有 near-plane 混合三角形；frame 173 的零 render 是所有三角形投影 bbox 都不进画面，不是 z-buffer 丢面。

held 分母是 source-frozen AABB 内、confidence 和连续法向 stencil 有效的 reference rays，不是全图像素。12 个分母为：

`[6102, 6071, 3611, 1788, 2234, 8366, 5643, 3280, 7884, 668, 4620, 3496]`

总计 53,763 rays；每幅图只占全图的 1.81%–22.69%。missing render 确实进入失败分母，没有“只在成功渲染处算 recall”的稀疏奖励 bug。存储的 atlas normals 与由 mesh 面积加权重算的 normals 基本一致，未发现 stale/mis-indexed normal。

独立复算的 macro 指标：

| arm | good-ray | joint depth+normal@20° | 官方点估计判定 |
|---|---:|---:|---|
| M0 | 0.205423 | 0.075578 | KILL（joint） |
| M1 | 0.211509 | 0.132790 | GO |
| M2 | 0.204325 | 0.139115 | GO |

绝对门槛为 good-ray 0.20、joint20 0.10。M2 的 good-ray 只高 0.004325，并且 12 个 held view 中 `seq1__frame00173.png` 完全没有渲染覆盖。

## P0：当前 seam gate 没有测到“同一重叠表面”的缝

当前实现对每条 frozen coverage edge 做的是：

1. 取两个 chart 的全部顶点；
2. 双向找最近顶点；
3. 保留距离不超过 1 m 的顶点；
4. 对这些顶点欧氏 NN 距离做 p50/p90，并要求每一条 edge 都满足 0.10 m/0.30 m。

这不是 frozen overlap correspondence，也不是 vertex-to-continuous-triangle surface distance。它会把视角采样错位的切向距离、只是在 1 m 内但并非同一物理表面的近邻，以及真正的 seam normal offset 混在一起。

证据非常明确：

- 单个 stride4 chart 内部 mesh edge 的 p50 已约为 0.363 m，高于 seam p50 门槛 0.10 m。
- 48 条 coverage edges 的 frozen symmetric overlap 只有 8.19%–24.72%，中位数 11.38%。
- M2 中被 1 m NN 规则称为“supported”的比例中位数却是 frozen overlap 的 2.21 倍；27/48 条 edge 超过两倍，说明 supported set 明显混入非 overlap 近邻。
- 把完全相同的当前 seam 算法用于 M0 source control，M0 也是 48/48 条 p50 失败、48/48 条 p90 失败：

| surface | pooled seam p50 | pooled seam p90 | per-edge p50 pass | per-edge p90 pass |
|---|---:|---:|---:|---:|
| M0 source control | 0.354980 m | 0.843857 m | 0/48 | 0/48 |
| M1 aligned | 0.347679 m | 0.844091 m | 0/48 | 0/48 |
| M2 aligned | 0.340962 m | 0.845709 m | 0/48 | 0/48 |

因此当前 seam 判定几乎是一个必杀项，并不能区分 M0、M1、M2 的真实 chart stitching 质量。尤其 M2 的 distortion 本来全部通过，它的 geometry KILL 完全来自这个 seam gate，所以 M2 和总体 `initializer_independent KILL` 都必须暂停解释。

正确修复应在 source-only 阶段、alignment 之前冻结每条 edge 的真实 overlap mask/correspondence，再在相同对应域上评估：

- sampling-aware point-to-continuous-triangle surface residual；
- normal-direction point-to-plane residual；
- 对应 normal angle/orientation；
- 每条 edge 的有效 support 和置信度。

修复不能从 aligned M1/M2 自己重新挑最近邻，否则仍会产生后验 cherry-picking。现有 GPU artifacts 可以全部复用，只需修改 CPU gate 后重评。

## M1 与 M2 当前可以分别下什么结论

### M1/DAV2

M1 即使移除所有 seam 项，也有独立的 chart area distortion KILL。去掉一个 global similarity scale 后：

- chart area ratio min/median/max = 1.103 / 1.410 / 2.747；
- 16 个 chart 中有 15 个高于 1.20 上限。

这不是单个 6-face 小 chart 的偶然异常，因此 M1 的 geometry KILL 可信。

### M2/MoGe3

M2 的 chart area ratio 为 0.867 / 1.050 / 1.172，face collapse/flip、normal flip、Jacobian 分位数等当前 distortion 项均通过。它在现有 held mapping diagnostic 上的 good-ray 与 joint20 点估计也通过绝对下限。

因此 M2 的正确状态是：`coverage point estimate GO; non-seam distortion GO; integrity pending corrected seam`，不是 final KILL，也不是 final GO。

## P1：absolute coverage 没有区间保证

正式报告只对相对差异做 paired block bootstrap；absolute coverage 使用 12-view macro 点估计直接过阈值。对四个 3-view 时间块枚举全部 4^4 个 equal-block bootstrap draw，得到：

| arm | metric | 点估计 | 95% block interval | floor |
|---|---|---:|---:|---:|
| M1 | good-ray | 0.211509 | [0.076509, 0.346510] | 0.20 |
| M1 | joint20 | 0.132790 | [0.049392, 0.221423] | 0.10 |
| M2 | good-ray | 0.204325 | [0.073147, 0.335503] | 0.20 |
| M2 | joint20 | 0.139115 | [0.050117, 0.228112] | 0.10 |

所以 “这 12 帧的点估计超过下限” 是可信的；“跨时间块的绝对 coverage 已正式过门” 不可信。未来若 `bounded_gate_eligible` 可能变成 GO，应要求 absolute lower confidence bound 过 floor，或明确把 absolute 判定降级为 deterministic sample-only diagnostic。

## 其他审计意见

- paired relative bootstrap 的左右方向、分块和 missing-ray 处理没有发现实现错误；它只能支持这个 12-view mapping sample 的相对非劣结论。
- distortion 中的 Jacobian 项目前用“aligned 和 initial 各自排序后的 singular values 逐项相除”，不是严格的 generalized deformation spectrum。用 initial tangent frame 重算后，当前 M1/M2 的 p05/p95 判定没有变化，因此这是 P1 方法严谨性问题，不改变本轮主结论。
- 当前报告已经正确声明 `production_eligible=false`。RADIO-UV、family retrieval 或 pose backend 不应在 corrected seam gate 和更充分 absolute coverage 验证前被接入生产结论。

## 最终可交付结论

- 可信：fresh exact16 数据链、严格 stride4 共拓扑、bounds、held 渲染数值；M1 distortion KILL；M1/M2 在这 12 帧上的 coverage 点估计。
- 暂停：M2 geometry KILL、双初始化都失败、总体 chart atlas 路线失败。
- 必做：修正并冻结 overlap correspondence/seam surface metric，用现有 artifact 做一次 CPU-only exact16 重评。
- 后续：只有 M2 corrected seam 通过且 absolute coverage 获得更多 route/block 支持后，才值得接 RADIO-UV 和 pose estimation。
