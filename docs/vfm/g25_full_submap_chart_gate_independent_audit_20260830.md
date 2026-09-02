# G25 完整子地图 chart gate 独立审计（2026-08-30）

## 裁决

当前 8-chart 结果只能保留为 **MAtCha 接入与显存可行性 smoke**，不能作为下列结论的科学证据：

- MoGe-3 initializer 优于 DAV2；
- alignment-only atlas 已经形成连续物理表面；
- chart atlas 优于 2DGS-derived planar map；
- 可以进入真实 query 位姿后端。

继续扩大实验是合理的，但下一轮必须先满足 source/held 上游隔离、overlap-aware source inventory、公共比较域以及 M0/M1/M2 同口径评估。任何一项缺失都只允许标记 `DIAGNOSTIC_ONLY`。

本审计没有读取 Cambridge query pose/GT 文件，也没有启动 GPU。

## 旧 8-chart 证据的硬问题

### P0-1：所谓 held pointmap 与 source pointmap 来自同一次全局 MASt3R run

旧 gate 使用的 `cameras.json` SHA256 为
`37a9fa646f8682a407224a926ae8af876f0e48e8e8eddce0b54c7037c159def1`，对应：

```text
/root/ICLPose-g24/output/g25_contextual/
  clean_room_geometry_ideal_pinhole_v5_exact_frozen_k_authority/
  fold0/mast3r_sfm/cameras.json
```

其冻结 authority 显示同一次 64-image、1000 coarse + 1000 refinement 的 MASt3R run 同时消费：

- seq4 的 19 个稀疏视图，包括 8 个 source 和后来称为 held 的 11 个视图；
- seq10、seq12、seq14 等其他路线视图。

因此 held seq4 pointmap 不是从 source chart 优化闭包中隔离的 reference，source pointmap 也可能经全局配对吸收 held 图像信息。旧 manifest 的 `route_clean=true` 只描述最终选中的名称，不证明上游计算闭包。旧 held 数字可以比较两个 arm 的工程行为，但不能解释为 held generalization。

### P0-2：均匀抽帧没有产生适合 chart alignment 的 overlap graph

8 个 source 的相邻相机基线为约 23--39 m，forward 方向差约 18--63 度。逐 chart 的跨-chart最近邻进一步暴露两个近似孤立 chart：

| Source chart | DAV2 cross-chart median | MoGe-3 cross-chart median |
|---|---:|---:|
| `seq4__frame00147.png` | 14.25 m | 14.81 m |
| `seq4__frame00183.png` | 7.42 m | 8.03 m |

把全局所有 chart 到任意其他 chart 的最近邻混在一起会同时惩罚合法的不重叠覆盖、奖励重复表面甚至表面塌缩。它不是物理 overlap consistency 指标。

### P0-3：M1/M2 的优化与导出支持域不同

旧实现使用每个 arm 自己的：

```text
initializer_valid AND MASt3R_reference_valid
```

最终 DAV2/MoGe-3 atlas 分别有 4,835/4,594 个顶点，逐 chart valid fraction 也不同。覆盖率、conditional depth error、nearest distance 和 normal 指标因此不是严格配对样本。MoGe-3 的小幅 near-overlap 增益可能部分来自支持域变化。

### P0-4：旧决策不是阈值驱动

`audit_goal_maplet_explicit_chart_geometry_gate.py` 的 `GO_TO_FULL_SUBMAP_GEOMETRY_GATE` 是固定字符串，不由预注册阈值或统计置信区间计算。旧 held evaluator 只报告 pooled-pixel conditional error；未渲染像素不进入误差，覆盖更少的方法可以通过只保留简单区域获得更好误差。

### P1：旧 MoGe-3 优势没有配对统计支持

对相同 11 个 held view 做 paired macro-view bootstrap，MoGe-3 减 DAV2 为：

| 指标 | 平均差 | 95% bootstrap CI | 判断 |
|---|---:|---:|---|
| coverage | +0.56 pp | [-0.43, +1.58] pp | 不显著 |
| per-view absolute-depth median | -0.085 m | [-0.340, +0.122] m | 不显著 |
| per-view absolute-depth P90 | +0.387 m | [+0.102, +0.678] m | MoGe-3 更差 |
| per-view relative-depth P90 | +0.012 | [+0.0006, +0.026] | MoGe-3 更差 |

这些 reference 本身仍有 P0-1 的泄漏，因此只能用于否定“旧结果已经证明 MoGe-3 占优”，不能用于决定最终 initializer。

## M0/M1/M2 的公平定义

需要同时报告两类实验，不能混成一张表。

### A. initializer-controlled 表

- M1：真实 DAV2 prior + MAtCha alignment-only；
- M2：focal-correct MoGe-3 + 同一 alignment-only；
- 完全相同的 model-neutral source chart inventory；
- 完全相同的 mapping poses、MASt3R source-only reference、1000 iterations、loss 和随机种子；
- 公共像素域为 `reference_valid AND dav2_valid AND moge3_valid`；
- 公共 stride-4/stride-8 face topology，不能让任一 arm 通过删点改变测量域。

若 keyframe selector 使用 MoGe-3 geometry，它只能作为 M2 system arm；不能用它来宣称 controlled M2 initializer 优于 M1。controlled 表应使用 source-only SfM/MASt3R visibility，或 DAV2/MoGe overlap 共识图选同一 inventory。

### B. system-level 表

- M0-surface：现有 2DGS finite surface，裁剪到同一物理 submap；
- M0-plane：由 M0 提取并真正序列化的 bounded planar side map；
- M1/M2：各自冻结的完整 atlas pipeline；
- 三者使用同一 mapping input pool、显式 held inventory、相机、reference mask、renderer 分辨率和评价 rays。

M0-plane 不能被当作无限平面；M1/M2 也不能额外获得 M0 没有的 query correspondence budget。建议再加两个 pose oracle：

1. 所有地图转换为 finite surface 2D--3D 后使用同一个 robust point-to-ray solver，隔离 carrier 能力；
2. 各 representation-native solver，评价完整表示 + solver 的系统上限。

## 上游与数据切分协议

1. 先冻结 source/held/forbidden image ID 和 route manifest，再运行任何重建。
2. source MASt3R 与 held-reference MASt3R 必须在不同、隔离输入根运行；不能先跑全集再删除文件。
3. authority 必须绑定 posed-COLMAP inventory、实际命令、代码、checkpoint、输入树和输出树 hash。
4. mapper 可以使用已知 mapping camera pose；不得读取 held/query pose 以外的 evaluator-only字段。报告中应写成 `uses_mapping_pose=true, uses_query_gt=false`，不要用含混的 `uses_ground_truth=false`。
5. held view 必须由独立冻结列表传入，不能用 `all route views - atlas chart names` 动态产生。
6. held view 只应覆盖预声明 submap 的可见范围；远离该 submap 的整条路线不能进入 completeness denominator。
7. selector 超参数只能在 source mapping views 上冻结。建议至少报告 overlap graph 对容差的稳定性。

对于 Cambridge 约 20--30 m 的常见深度，selector 默认的
`max(1 m, 10% depth)`、60 度 normal、5% overlap、40 m maximum baseline 很宽松，可能将 2--3 m depth disagreement 或错误平行表面接成边。需要加入：

- baseline / overlap median depth；
- 最小与最大 triangulation/parallax angle；
- camera forward-angle；
- 双向 depth-order/visibility；
- 例如 0.5/1.0 m、5/10%、30/45/60 度的预注册稳定性表。

## 几何主指标

### 1. Held ray completeness--accuracy

主指标必须把 missing surface 当失败：

```text
good_ray_recall =
  # reference-valid rays with rendered surface and
    |z_render-z_ref| <= max(0.5 m, 0.05*z_ref)
  / # reference-valid rays
```

同时报告：

- macro-view coverage；
- metric depth median/P90 和 AbsRel median/P90；
- depth+normal joint recall，normal thresholds 10/20/30 度；
- depth/normal boundary F1（2/4 pixels）；
- 每 view 数值、temporal-block macro 和 paired bootstrap CI；
- pooled pixel 只作为附表。

MASt3R reference 不是 sensor GT，因此这些指标只能称 mapping proxy。最终“更适合定位”的结论必须由 pose oracle / real matcher 给出。

### 2. 预冻结 overlap-edge consistency

只在 selector 冻结的共视边上报告：

- symmetric point-to-triangle distance；
- point-to-plane signed distance / surface thickness；
- signed normal 与 unsigned normal angle；
- overlap area、seam length 和 occlusion disagreement；
- 每条边和每个 connected component 的 macro 统计。

不能继续使用“每个顶点到任意其他 chart 的最近点”作为主指标。

### 3. 防塌缩与 canonicalization

- per-chart aligned/initial triangle area ratio P5/P50/P95；
- local Jacobian singular-value ratio、flipped-face fraction；
- aligned deformation magnitude；
- duplicate-layer thickness；
- canonical family 数、source-view 数/family、small-component area fraction；
- largest connected physical surface area fraction；
- source view ID 不得成为在线 candidate identity。

### 4. 资源

旧 `0.2 MiB versus 49.97 MiB` 不可比：前者只有 8 个低采样 chart，后者是全场景 509,572 primitives 和 hierarchy。

应序列化同一 submap 的部署包并分别报告：

- geometry-only bytes；
- vertices/faces/UV/normals/confidence/family graph；
- RADIO field bytes及其维度、量化；
- GPU resident bytes；
- 相同 renderer、256x144、batch-1 的 warm median/P90 latency；
- map build wall time和 source image count。

若每个 36x64 texel 保存 1280-D FP16 RADIO，单 chart 约 5.6 MiB，feature field 很快比 geometry 更大。必须先 canonical family fusion，再考虑 64--128-D projection/quantization或稀疏多分辨率 field。

## 建议预注册晋级条件

以下是进入 chart correspondence / pose oracle 的建议硬门，不是从旧结果反推的阈值：

1. lineage：source/held 计算闭包零重叠，M0/M1/M2 common inventory/domain 全部 hash replay；
2. M2 对 M0 的 macro good-ray recall 非劣 95% CI 下界不低于 -2 pp；
3. 冻结 overlap edges 上 seam point-to-plane P50/P90 不高于 0.10/0.30 m；
4. normal angle P50/P90 不高于 10/30 度，且 signed flip rate 单独通过；
5. 无明显 area collapse、face flip 或 duplicate layer；
6. canonical family 后的完整部署包（含 RADIO）内存和 render latency不劣于 M0；
7. GT-submap/common correspondence pose oracle 不低于 M0，并在沿大平面方向的 capture 上有明确增益。

若只通过 1--5，可进入 pose oracle，不得替换主地图。若 pose oracle 不低于 M0但真实 RADIO matching 失败，问题归于 feature/matcher；若 pose oracle 已低于 M0，问题归于 geometry/carrier，不应再训练 matcher 掩盖它。

## 资源瓶颈预估

- chart alignment/matching 近似随 chart count 平方增长；8 charts/1000 steps 约 86 s，完整场景不能直接按 46--64 charts 一次运行；建议一个 connected 12--16 chart facade gate；
- Python CPU triangle rasterizer会随 held views x faces 线性变慢，完整 gate 应改为同一 GPU/矢量化 renderer 后再比较 M0/M1/M2 latency；
- exact all-other-chart KD-tree 同样不应扩到全场景，且它在科学上不是正确 overlap metric；
- MoGe-3 初始化显存约 2.8--3 GB，不是当前主瓶颈；主要瓶颈是 pairwise alignment、canonicalization 与 RADIO UV field 内存。

## 当前最小正确执行顺序

1. 冻结 source-only selector inventory和显式 held inventory；
2. 分离重建并生成可验证的 disjoint upstream authority；
3. 生成公共 DAV2/MoGe/reference comparison domain；
4. 跑 controlled M1/M2 12--16 chart gate；
5. 用同一 held rays 增加 M0-surface/M0-plane；
6. 只有 geometry + resource gate 通过后，构建 canonical surface families；
7. 再接 RADIO UV 和 GT-submap chart correspondence / pose capture oracle；
8. real query matching最后运行，Gaussian refinement仅在 M2 geometry 被明确证明受 alignment-only 上限限制时作为离线上界。

