可以，**fine offset 很可能就是当前瓶颈的核心组成部分之一**。但我会把它说得更精确一点：现在的问题大概率不是单纯“PnP 错了”，也不是单纯“VFM 不行”，而是：

> **render-pose-conditioned 3D lifting + fine offset 没有把 render side 移到真实物理对应点 + query/render offset 使用目标不一致**
> 这三件事叠加后，导致 PnP 只能复现当前 perturbed render pose，形成 render-lock。

我下面完全按 **pose refinement** 主线来梳理，不再把目标转向 hypothesis verification。以下判断基于我对最新 `vfm` 分支可见代码、最新 commit、MATCHA coarse-to-fine、rendered keypoint matching、eval/diagnostic 工具和你 repo 里 perturbation plan 的静态阅读；我没有跑你的数据，所以具体数值层面的因果还需要用下面的诊断实验确认。

---

## 1. 重新判断：fine offset 确实很可疑，但它和 render-lock 是同一个问题的两面

你当前 PnP 输入的 3D 点是这样来的：

```text
query_xy
  ↕ feature match
render_xy
  ↓ sample rendered_depth(render_xy)
depth + render_pose_w2c + render_camera
  ↓ backproject
world_xyz
  ↓ PnP(query_xy ↔ world_xyz)
estimated_query_pose
```

代码里 `keypoint_feature_matches_to_pnp_matches()` 会在 `match.render_xy` 上采样 rendered depth，然后用 `backproject_depth_to_world(..., render_pose_w2c)` 生成 `xyz`，最后构造 `QueryTo3DMatch(xy=match.query_xy, track_id=match.render_index, xyz=...)`。这说明 **render_xy 的位置直接决定了被 PnP 使用的 3D 点**；如果 render_xy 没有移动到真实物理对应点，PnP 的 3D anchor 就会被当前 render pose 锁住。([GitHub][1])

所以问题可以写成：

[
X_i = T_r^{-1}(d_i K^{-1} \bar u^r_i)
]

[
\hat T_q = \operatorname{PnP}(u^q_i, X_i)
]

其中 (T_r) 是 perturbed render pose，(u^r_i) 是匹配得到的 render side 像素。如果 fine matching 没有让 (u^r_i) 对应到 query 里的同一个真实 3D 表面点，那么 (X_i) 本身就是一个由 (T_r) 定义出来的“自洽 3D 点”。这时 PnP 最容易得到：

[
\hat T_q \approx T_r
]

于是你看到：

```text
final pose error ≈ render pose perturbation error
```

这和你观察完全一致。

---

## 2. query-side offset 和 render-side offset 的作用不一样

这里是我认为你现在最需要重新区分的点。

### 2.1 query-side offset 主要改变 2D measurement

query-side offset 修改的是：

```text
PnP 左边的 2D observation: query_xy
```

它可能在 GT render 对齐时有帮助，因为它能把 query token/cell center 调整到更像 keypoint 的位置。但在 perturbed render refinement 里，它不一定能提供正确的 pose correction，甚至可能把 residual 吃掉。

也就是说，query-side offset 可能让 descriptor matching 看起来更顺，但它没有改变：

```text
render_depth 被采样的位置
world_xyz 的来源
```

所以如果真正错误在“render side 没有采到正确 3D 表面点”，query-side offset 是修不了的。

---

### 2.2 render-side offset 才会改变 PnP 的 3D 点

render-side offset 修改的是：

```text
render_xy
render_depth(render_xy)
world_xyz(render_xy, render_pose)
```

这才是 pose refinement 里最关键的东西。你的 repo 里的 IF-A plan 也把这个假设写得很明确：query-side fine offset 可能不是 render-conditioned depth 的正确修正；render-side offset 会改变 sampled depth 和 3D point，因此可能恢复 metric correspondence。([GitHub][2])

所以目前最核心的判断是：

> **如果 render-side fine offset 没有学会或没有足够搜索空间移动到真实对应的 render pixel，那么 PnP 就几乎必然锁在 render pose。**

---

### 2.3 both-side offset 也可能有副作用

你目前有 `query_offset_logits` 和 `render_offset_logits`，`apply_offset_logits_to_matches()` 会把 coarse cell center 移到 8×8 offset bin 的中心；注释里也说明它只改几何位置，不改 descriptor similarity/confidence。([GitHub][3])

如果同时开 query-side 和 render-side offset，有一个风险：

```text
query_xy 被移动
render_xy 也被移动
descriptor/local correlation 变好
但真实 pose residual 被抵消或变弱
```

这种情况下，PnP 的输入看起来更“自洽”，但并没有更接近 GT pose。

所以短期诊断里，不要只看 final pose error。必须分开看：

```text
query-side offset 是否让 query reprojection 更接近 GT？
render-side offset 是否让 sampled XYZ 更接近真实 3D point？
both-side offset 是否只是让 feature score 变高，但 pose correction 变差？
```

---

## 3. coarse cell fine offset 搜索空间确实可能不够大

`matcha_coarse_to_fine_keypoint_matches()` 的默认 `fine_search_radius_px=8.0`，先做 coarse dual-softmax，再应用 offset logits，然后对 render side 做 local refinement 或 attention/fine modes。([GitHub][3])

8 px 对 GT render 或极小扰动可能够，但对 25 cm translation、1° rotation 或近景结构很可能不够。更重要的是：**8×8 pair fine offset 只是在当前 coarse cell 内选 sub-cell 坐标**，它不能跨 cell 找真实对应点。`_offset_label_to_xy()` 明确是根据当前 `cell_index` 和 8×8 bin 生成 cell 内坐标。([GitHub][3])

所以存在两个 search-space failure：

### failure 1：真实对应点仍在同一 coarse cell，但 sub-cell 预测不准

这时 render-side pair fine head 可以解决，但前提是它被 render-side 监督训练过。

### failure 2：真实对应点已经跑到相邻或更远 coarse cell

这时 8×8 fine offset 不可能解决，因为它只能在原 coarse cell 内移动。你需要 `expand_matches_with_render_local_offsets()` 这种跨 render cell 的候选扩展；这个函数当前已经存在，作用正是“query xy 不动，render xy 在邻近 render cell/subcell 内扩展，并且让后续 depth backprojection 采到不同 3D 点”。([GitHub][3])

这点和你的直觉完全一致：**fine offset 搜索空间不足会让系统陷入局部最优，而这个局部最优通常就是当前 render pose。**

---

## 4. 目前实现和方法里的主要问题排序

### P0：PnP 的 3D anchor 依赖 perturbed render pose

这是最高优先级问题。

只要 `world_xyz` 是由：

```text
render_xy + rendered_depth + render_pose_w2c
```

临时反投影出来的，系统就强依赖 render-side matching 的正确性。GT render 成功并不能证明这个机制能 refinement；它只证明 render pose 已经正确时，query/render feature 能对齐。

这不是要求你立刻改成完全 persistent landmark map，但你至少要在诊断里把问题拆开：

```text
PnP geometry 是否能修正 pose？
render-side matching 是否能找到正确 3D 表面？
fine offset 是否能把 render_xy 移到正确位置？
```

---

### P0：当前 fine offset 对 +25 cm 几乎没有有效修正信号

你 repo 里的 plan 记录了一个非常关键的 baseline：

```text
GT full182: median 0.0187m, S@10 1.0, S@25 1.0
+25cm full182: median 0.2432m, S@10 0.011, S@25 0.731
+25cm fine offset: 12.74px -> 12.71px, improved ratio 0.419
confidence inlier/outlier gap: 0.0029
```

这个 `12.74px -> 12.71px` 基本等于 fine offset 没有修正几何偏移；`0.0029` 的 confidence gap 也说明当前 confidence 几乎不能区分 pose-usable match 和 bad match。([GitHub][2])

这组数据非常支持你的怀疑：**fine offset/匹配置信度目前没有给 PnP 足够的 correction signal。**

---

### P0：render-side learned fine head 可能没有被正确训练

`apply_pair_fine_logits_to_matches()` 默认 `target_side="render"`，但你的 plan 里明确写到：当前 checkpoint 可能是为 query-side pair fine 优化的；render-side head 需要显式训练，因为 render-side offset 会影响 depth 和 3D point selection。([GitHub][3])

所以我认为要重点检查这几个组合：

```text
matcha_pair_fine_side=query
matcha_pair_fine_side=render
matcha_pair_fine_side=none
matcha_cell_offset_side=query
matcha_cell_offset_side=render
matcha_cell_offset_side=both
matcha_cell_offset_side=none
```

尤其要避免一种情况：

```text
用 query-side 监督训练出来的 fine logits
却在 render side 上解释成 8×8 render subcell
```

这会产生看似合法、但几何语义错误的 render_xy。

---

### P1：local correlation / softargmax 容易被局部最优困住

`refine_render_keypoint_matches_by_local_correlation()` 默认 `search_radius_px=4.0`，候选点是围绕当前 `match.render_xy` 加局部 offsets；如果多个候选分数接近，它会 tie-break 到离原始 render_xy 最近的位置。([GitHub][1])

这对稳定微调是合理的，但对 pose refinement 可能有副作用：

```text
当前 render pose 偏了
真实对应点在 local window 外
VFM similarity 又比较平滑
argmax/softargmax 会停在原位置附近
render_xy 几乎不动
sampled depth / xyz 几乎不变
PnP 回到 render pose
```

bilateral refinement 也有类似风险，因为它同时移动 query 和 render；代码里 bilateral mode 会同时在 query/render 两边搜索，argmax tie-break 也偏向更近的 query/render 候选。([GitHub][3])

---

### P1：coarse-level dedup 和 `track_id=render_index` 的语义不够稳定

`QueryTo3DMatch.track_id` 当前来自 `match.render_index`。这在单次 render 内可以当作去重 index，但它不是稳定 3D landmark ID。([GitHub][1])

这会影响：

```text
deduplication
RANSAC sampling diversity
match diagnostics
跨 render pose 的 identity 一致性
```

短期不一定要完全重构，但建议至少改名或新增字段区分：

```text
render_cell_id
render_candidate_id
stable_anchor_id
```

不要把 `render_index` 当成真正 3D track id。

---

### P1：`max_render_depth_delta_m` 有潜在 guard bug

在 `keypoint_feature_matches_to_pnp_matches()` 里，`center_depth` 只有 `has_grid=True` 时才会被计算；但 `max_render_depth_delta_m` 的 guard 会直接用 `np.isfinite(center_depth)`。如果某些调用传了 `max_render_depth_delta_m` 但没有传 `render_grid_width/render_grid_height`，这里逻辑上会出问题。([GitHub][1])

建议直接改成强约束：

```python
if max_render_depth_delta_m is not None and float(max_render_depth_delta_m) >= 0.0:
    if not has_grid or center_depth is None:
        raise ValueError(
            "max_render_depth_delta_m requires render_grid_width and render_grid_height"
        )
    guard_valid &= (
        np.isfinite(center_depth)
        & (np.abs(depth_values - center_depth) <= float(max_render_depth_delta_m))
    )
```

---

### P1：camera / canvas / feature-map 坐标系必须重新做 round-trip

你的 eval 里有 `_render_canvas_camera_from_base()`，它创建更大 render canvas 时不缩放 focal，只平移 principal point。这个逻辑本身可以是对的，但只要 render RGB、render depth、render feature、render_xy、backprojection camera、PnP camera 有任何一个用错坐标系，就会产生系统性误差。([GitHub][4])

尤其是以下组合很容易混：

```text
render_width/render_height
canvas_width/canvas_height
base_camera intrinsics
canvas_camera intrinsics
depth map resolution
feature map resolution
query image coordinate
render image coordinate
```

建议把 camera/depth/feature round-trip 作为 P0 级测试，而不是后面再查。

---

### P2：confidence 目标不是 pose-usable

当前 plan 已经指出 confidence inlier/outlier gap 只有 `0.0029`，并且 IF-F 计划把 confidence label 改成 pose-usable：GT reprojection error ≤ 8px、alpha valid、depth-edge safe 为正；>24px 或 visibility invalid 为负；8–24px ignore。([GitHub][2])

我认为这个方向很重要。因为现在不是“match 多不多”的问题，而是：

```text
哪些 match 能真正让 PnP 离开 render pose？
```

confidence 如果只是 descriptor similarity 或 patch correctness，它不一定和 PnP usefulness 一致。

---

### P2：训练数据需要 perturb-aware，而不是 GT-aligned only

你 repo 里的 perturbation-aware adapter plan 已经提出加入 `A_gt`, `B_trans025`, `C_trans050`, `D_reference` 这些 pair types，并在 joint cache 里保存 perturb translation/rotation metadata、no-match/ignore labels、hard false matches 和 iterative update eval。([GitHub][5])

这对 pose refinement 是必要的。GT-aligned 训练会让模型学会：

```text
query/render 已经对齐时，哪个 offset 好
```

但 pose refinement 需要模型学会：

```text
render pose 偏了以后，render side 应该往哪里移动，才能采到正确 3D 点
```

这两个任务不是同一个任务。

---

### P2：cache keys 最新 commit 已经动过，但仍建议强制校验

最新 `vfm` commit 是 2026-06-10 的 “Fix render-side defaults and cache keys for MATCHA eval”。([GitHub][6])

这说明你已经意识到 cache key 和 render-side defaults 的风险。即便如此，我建议在所有 perturbation 实验里继续强制写入 manifest：

```text
render_pose_hash
render_camera_hash
render_rgb_hash
render_depth_hash
render_feature_hash
radio/model checkpoint hash
adapter/joint checkpoint hash
render_width/height
canvas_width/height
gaussian ply hash or mtime
```

这个问题一旦存在，会制造非常像“方法坏了”的假象。

---

## 5. 具体诊断计划：先证明瓶颈在哪里

我建议按下面顺序做。目标不是多跑实验，而是每一步都能排除一类根因。

---

### Step 0：固定实验基线和日志字段

先固定一个小规模 q32 +25cm benchmark，再跑 full182。你的 plan 里已有 promotion gate：q32 +25cm median t 至少提升 3cm 或 S@25 提升 10% absolute；full182 +25cm 目标是 median ≤ 0.20m 或 S@25 ≥ 0.85，且 GT render S@10 不低于 0.98。([GitHub][2])

每个 query 记录这些字段：

```text
query_id
render_pose_t_error
render_pose_R_error
final_pose_t_error
final_pose_R_error
final_minus_render_t
final_minus_render_R
num_raw_matches
num_pnp_matches
num_pnp_inliers
median_reproj_error
pnp_residual_at_render_pose
pnp_residual_at_final_pose
median_query_offset_norm
median_render_offset_norm
median_render_xy_to_oracle_px
fine_capture_rate_8px
fine_capture_rate_16px
fine_capture_rate_32px
confidence_inlier_outlier_gap
```

你已有 `_match_table_rows_for_query()` 导出 query/render/world/similarity/confidence/GT reprojection/inlier/patch offset 等 match-level diagnostics，可以在这个基础上加 `oracle_render_xy`、`render_xy_error_to_oracle`、`offset_before/after`。([GitHub][4])

---

### Step 1：几何 sanity check，排除 PnP/depth/camera bug

先不要管 VFM。

做三个 oracle：

#### 1.1 same-pose round-trip

对每个 render match 或随机 depth pixel：

```text
render_xy
depth = rendered_depth(render_xy)
X = backproject(render_xy, depth, render_camera, render_pose)
render_xy_reproj = project(X, render_camera, render_pose)
```

要求：

```text
median roundtrip error < 0.1 px
p95 < 0.5 px
```

如果这里不过，优先查：

```text
depth 是否是 camera z-depth
camera intrinsics 是否对应 depth resolution
canvas camera 是否用错
pose convention 是否 w2c/c2w 反了
```

#### 1.2 synthetic render-lock test

人为构造：

```text
query_xy = render_xy
X = backproject(render_xy, depth, perturbed_render_pose)
PnP(query_xy, X)
```

预期：

```text
PnP ≈ perturbed_render_pose
```

如果成立，就证明当前结构天然会 lock；这不是 bug，而是 formulation 的退化模式。

#### 1.3 GT-correct correspondence oracle

用 GT pose 构造真实对应：

```text
真实 3D 点 X
query_xy = project(X, GT_query_pose)
render_xy = project(X, perturbed_render_pose)
depth = rendered_depth(render_xy)
PnP(query_xy, X 或 backproject(render_xy, depth, perturbed_render_pose))
```

预期：

```text
PnP 应该能拉回 GT
```

你的 IF-C plan 也要求用 GT-correct matches 检查：如果正确 correspondences 下 +25cm 都拉不回 GT，那就是 geometry/depth/camera 错；如果能拉回，那瓶颈在 matching/fine offset。([GitHub][2])

---

### Step 2：量化 perturbation 导致的真实 optical flow 是否超出搜索空间

你已经有 `diagnose_render_pose_perturbation.py`，其中 `_project_flow_stats()` 会把 GT-render depth points 投影到 perturbed camera，并统计 pixel displacement 的 median/p90/p95。([GitHub][7])

把它用于：

```text
+5cm
+10cm
+25cm
+50cm
yaw/pitch/roll 0.25°, 0.5°, 1°, 2°
```

然后和当前设置比较：

```text
fine_search_radius_px = 4 / 8 / 16 / 32
coarse cell width/height
render_side_local_offset_radius_cells = 0 / 1 / 2
```

必须得到一个表：

```text
perturbation | flow_median | flow_p90 | flow_p95 | within_same_cell | within_8px | within_16px | within_cell_radius_1 | within_cell_radius_2
```

如果 `flow_p90 > 8px`，那默认 fine radius=8 根本不是可靠 refinement，只是局部微调。
如果 `within_same_cell` 很低，8×8 pair fine 也无能为力，必须做跨 cell render-side candidate expansion。

---

### Step 3：直接评估 fine offset 是否朝正确方向移动

对每个 match 计算 oracle render coordinate：

```text
oracle_render_xy = project(true_world_point, perturbed_render_pose)
```

或者如果没有 stable true_world_point，就用 GT pose/depth 近似生成可见 surface correspondence。

然后对比：

```text
render_xy_before_offset_error = ||render_xy_before - oracle_render_xy||
render_xy_after_cell_offset_error
render_xy_after_pair_fine_error
render_xy_after_local_corr_error
```

分开统计：

```text
query cell offset only
render cell offset only
both cell offset
pair fine query
pair fine render
local corr argmax
local corr softargmax
fine attention
bilateral
```

你现在已有的 `+25cm fine offset: 12.74px -> 12.71px` 说明当前 fine offset 几乎没有把点往 oracle 方向推。下一步要看到的是：

```text
render-side before -> after 至少有明显下降
例如 12.7px -> 7px / 5px / 3px
```

否则不应该期待 PnP 有明显改善。

---

### Step 4：做 offset-side ablation，锁定 query/render/both 的真实作用

建议先跑 q32 +25cm：

| 实验 | query cell offset | render cell offset | pair fine side |                local search | 目的                                    |
| -- | ----------------: | -----------------: | -------------- | --------------------------: | ------------------------------------- |
| A0 |               off |                off | none           |                           0 | coarse baseline                       |
| A1 |                on |                off | none           |                         0/8 | query offset 是否有用                     |
| A2 |               off |                 on | none           |                         0/8 | render cell offset 是否有用               |
| A3 |                on |                 on | none           |                         0/8 | both 是否互相抵消                           |
| A4 |               off |                off | query          |                         0/8 | query pair fine 是否只是移动 2D measurement |
| A5 |               off |                off | render         |                         0/8 | render pair fine 是否真的采到更好 3D          |
| A6 |               off |                 on | render         |                        8/16 | render side full path                 |
| A7 |                on |                 on | query          |                        8/16 | 当前强配置复现                               |
| A8 |               off |                off | none           | render-side cell radius 1/2 | 搜索空间是否主瓶颈                             |

关键指标不是只看 final median，而是：

```text
final_pose_error < render_pose_error 的比例
median(final_error - render_error)
median(||render_xy - oracle_render_xy||) before/after
PnP-inlier GT@16/32
confidence gap
spatial coverage
```

如果 A8 明显提升，说明主要问题是 search range/candidate support。
如果 A5/A6 提升，说明 render-side fine head 有信号。
如果 A1/A4 提升但 PnP 仍 lock，要警惕 query-side offset 只是改善表面 residual，没有改对 3D。
如果 A3/A7 变差，说明 both-side offset 互相污染。

---

### Step 5：启用 render-side local candidate expansion，但不要只取最高 feature score

`expand_matches_with_render_local_offsets()` 的正确使用位置应该是：

```text
coarse matching
→ optional query/render cell offset
→ render-side local cell expansion
→ 对每个 expanded render_xy sample depth/backproject
→ PnP/RANSAC 或 geometry-aware scoring
```

plan 里也写了：radius > 0 时，在 coarse matching 后、`keypoint_feature_matches_to_pnp_matches()` 前调用 expansion。([GitHub][2])

重要的是：不要只按 descriptor similarity 选 top1 expanded candidate。因为 VFM similarity 在重复结构上可能错。更合理的是：

```text
每个 coarse match 保留 top-K render candidates
每个 candidate 有:
  render_xy
  render_depth
  alpha
  depth_gradient
  descriptor_score
  confidence
  local offset norm
然后交给 RANSAC / PnP 选择几何一致的组合
```

也就是说，render-side expansion 应该把问题从：

```text
每个 query 只有一个可能 3D 点
```

变成：

```text
每个 query 有 K 个可能 3D 点，PnP/RANSAC 通过全局几何一致性选
```

这能显著降低局部 feature argmax 错误对最终 pose 的支配。

---

## 6. 具体优化计划：从最小改动到中期重训

### Phase 1：一两天内能完成的代码级修复和诊断

优先做这些，不需要重训。

#### 1. 修 `max_render_depth_delta_m` guard

避免没有 grid 时使用 `center_depth=None`。

#### 2. 加强 cache manifest 校验

最新 commit 已经涉及 cache key 修复，但 perturbation eval 仍建议在每个 output 里写一个 `render_cache_manifest.csv/jsonl`：

```text
query_id
render_pose_hash
render_camera_hash
render_rgb_cache_path
render_depth_cache_path
render_feature_cache_path
hit_or_miss
rgb_sha1
depth_sha1
feature_sha1
checkpoint_sha1
```

#### 3. 加 same-pose round-trip test

测试：

```text
backproject(render_xy, depth, render_camera, render_pose)
project(X, render_camera, render_pose)
```

这是定位 camera/depth/canvas 错误最快的测试。

#### 4. 加 synthetic lock test

验证当前 structure 是否必然返回 render pose。这个实验一旦成立，后面的目标就清楚了：必须让 render_xy 产生正确 3D 点，或者引入 stable 3D anchor。

#### 5. 加 flow capture report

直接调用 `_project_flow_stats()` 统计 `flow_p90/p95`，并输出：

```text
flow_p90_px
flow_p95_px
fine_radius_px
cell_radius_needed
within_current_cell_rate
within_radius8_rate
within_radius16_rate
within_radius32_rate
```

---

### Phase 2：不重训的搜索空间实验

这部分直接测试你的 fine offset 假设。

#### 2.1 render-side local expansion sweep

跑：

```text
render_side_local_offset_radius_cells = 0, 1, 2
max_candidates_per_match = 1, 9, 25
fine_search_radius_px = 0, 8, 16, 32
```

建议先用：

```text
query xy 固定
render xy 扩展
pair_fine_side = none 或 render
cell_offset_side = render 或 none
```

不要一开始就 both-side 全开，否则不容易归因。

#### 2.2 local refinement mode sweep

比较：

```text
argmax
softargmax
fine_attention_argmax
fine_attention_softargmax
bilateral_argmax
bilateral_softargmax
```

我预期：

```text
fine_attention_* 可能比 bilateral 更适合 PnP
```

因为代码注释里也明确 fine attention 保持 query measurement fixed，避免 query-side offset noise 注入 PnP，同时用 query local context 消解 render-side local search。([GitHub][3])

#### 2.3 PnP 前增加 spatial/depth coverage filter

不要只靠 match confidence。增加：

```text
coverage_filter_grid = 8 或 12
coverage_filter_max_per_cell = 4/8
min_depth_range
min_image_coverage
max_matches_per_render_cell
max_matches_per_depth_bin
```

目标是避免 PnP 被一小块墙面/窗户/重复纹理支配。

---

### Phase 3：训练 render-side fine head

如果 Phase 2 证明 render-side search 有 signal，就开始重训。

训练目标要从：

```text
GT-aligned query/render sub-cell correctness
```

改成：

```text
perturbed render 下，render side 应该移动到哪个 sub-cell / neighboring cell 才能采到 query 对应的 3D point
```

你已有 IF-B plan：加入 `--render_pair_fine_loss_weight` 和 `--query_pair_fine_loss_weight`，确保 `render_pair_fine_loss_weight > 0` 时用 render offset soft labels 训练 render-side pair fine head。([GitHub][2])

建议训练配置：

```text
freeze descriptor backbone / projection head
只训练:
  pair_fine_head
  confidence_head
  optional detector/reliability head

pair types:
  A_gt
  B_trans025
  C_trans050
  D_reference 或 retrieval-like pose
```

loss 建议：

```text
L = L_coarse_match
  + λ_render_fine * CE/softCE(render_offset_label)
  + λ_query_fine * CE/softCE(query_offset_label, 小权重或关闭)
  + λ_conf * BCE(pose_usable_label)
  + λ_false * hard_negative_margin
```

我的建议是先这样设：

```text
render_pair_fine_loss_weight = 1.0
query_pair_fine_loss_weight = 0.0 或 0.25
confidence target = pose-usable
descriptor lr = 0 或很小
head lr = 正常
```

不要先让 descriptor 大幅更新，否则你很难判断问题是 fine head、confidence 还是 feature space 变化导致的。

---

### Phase 4：confidence retargeting

当前 confidence gap `0.0029` 太小，基本没法用于 PnP 选择。建议把 confidence label 改成：

```text
positive:
  GT reprojection error <= 8px
  alpha valid
  depth edge safe
  render-side offset not clipped
  optional: local depth variance low

negative:
  GT reprojection error > 24px
  visibility invalid
  alpha invalid
  depth edge unsafe
  hard false match

ignore:
  8px < error <= 24px
  ambiguous visibility
```

这和你 IF-F plan 一致。([GitHub][2])

PnP 时排序/筛选用：

```text
score = confidence_pose_usable
      * descriptor_score
      * alpha_weight
      * depth_edge_weight
      * coverage_weight
```

而不是单纯 dual-softmax confidence。

---

### Phase 5：从单点 match 改成多候选 2D-to-3D PnP

这是解决 render-lock 的一个关键结构改动，但仍然属于 pose refinement，不是 hypothesis verification。

现在每个 query match 大概对应一个 3D：

```text
query_i -> render_i -> X_i
```

建议改成：

```text
query_i -> {render_i1, render_i2, ..., render_iK}
        -> {X_i1, X_i2, ..., X_iK}
```

然后 PnP/RANSAC 每轮 sampling 时允许从每个 query 的候选 3D 里选一个，或者先把所有候选放进去但加约束：

```text
同一个 query token 最多一个 inlier
同一个 render cell / anchor 最多一个 inlier
inliers 需要 spatial coverage
```

这样可以避免错误 top1 render fine offset 直接毁掉一个 match。

---

## 7. 成功/失败判断标准

我建议你不要用“final error 是否略微下降”作为唯一判断，而是设下面这些 gate。

### Gate 1：geometry oracle 必须过

```text
GT-correct matches + +25cm render pose
PnP/residual solver 应该能回到接近 GT
```

如果不过，查 camera/depth/pose convention。

---

### Gate 2：fine offset 必须显著降低 render-side oracle error

当前已知是：

```text
12.74px -> 12.71px
```

这个不够。至少希望 q32 +25cm 看到：

```text
median render_xy oracle error 下降 > 3px
或者 improved ratio > 0.65
或者 clipped-at-boundary rate 明显下降
```

如果扩展 radius=1/2 后才改善，说明 search range 是主因。
如果扩展后也不改善，说明 descriptor/fine head 没有判别力，需要重训/换监督。

---

### Gate 3：PnP 必须离开 render pose，并且方向正确

记录：

```text
render_error_to_gt
final_error_to_gt
final_delta_from_render
```

希望看到：

```text
final_error_to_gt < render_error_to_gt
final_delta_from_render 不再接近 0
final_delta_from_render 方向与 GT correction 一致
```

不是所有样本都要成功，但 median 上必须明显离开 render lock。

---

### Gate 4：confidence gap 必须变大

当前 gap 是 `0.0029`。短期目标可以设：

```text
confidence inlier/outlier gap > 0.03
```

这也是你 plan 里的 IF-F success criterion。([GitHub][2])

---

### Gate 5：GT render 不回退

你目前 GT render 很强：

```text
median 0.0187m
S@10 1.0
S@25 1.0
```

所有 perturbation 优化都要守住：

```text
GT S@10 >= 0.98
PnP-inlier GT@16 不下降超过 2% absolute
```

你的 plan 里也已经有类似 gate。([GitHub][2])

---

## 8. 我建议的最终优先级

按性价比排序：

1. **先做 geometry oracle / round-trip / synthetic lock**
   这一步确认到底是不是 PnP/camera/depth bug。

2. **用 `_project_flow_stats()` 测真实 flow，判断 8px / 当前 cell 是否够**
   如果 p90/p95 超出，就不要再期待当前 fine offset 能修。

3. **跑 offset-side ablation：query-only / render-only / both / none**
   重点看 render-side oracle error 和 final-minus-render delta。

4. **启用 render-side local cell expansion，query xy 固定**
   先不重训，直接看 search space 是否有 signal。

5. **如果 expansion 有 signal，训练 render-side pair fine head**
   不要用 query-side fine head 代替 render-side fine head。

6. **把 confidence 改成 pose-usable target**
   解决 PnP 选不出好 match 的问题。

7. **中期改成 per-query multi-candidate 2D-to-3D PnP**
   避免 top1 render fine offset 错误把 PnP 绑死。

---

## 9. 最后给一个压缩版判断

你现在的瓶颈很可能是：

```text
GT render:
  query/render 已对齐
  coarse/fine offset 不需要真正恢复大位移
  render_depth backprojection 得到正确 3D
  PnP 很准

perturbed render:
  正确对应点需要 render-side 位移
  当前 fine offset 搜索空间/监督/side 使用不够
  render_xy 没移动到真实物理对应点
  sampled depth + render_pose 生成了自洽但错误的 3D
  PnP 返回接近 render pose
```

所以接下来不要优先调 PnP 参数，也不要笼统地调 VFM feature。最应该做的是：

```text
证明 render-side fine offset 是否能找到正确 3D 点；
证明当前搜索空间是否覆盖真实 perturbation flow；
证明 query-side offset 有没有在掩盖 residual；
证明 confidence 是否能选择 pose-usable matches。
```

如果这四件事修通，当前 pose refinement 主线才有机会摆脱“final pose ≈ perturbed render pose”的瓶颈。

[1]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/vfm/rendered_keypoint_matching.py "ICLPose/feature_extract/vfm/rendered_keypoint_matching.py at vfm · Arthurshen926/ICLPose · GitHub"
[2]: https://github.com/Arthurshen926/ICLPose/blob/vfm/docs/superpowers/plans/2026-06-10-render-perturbation-if-lines.md "ICLPose/docs/superpowers/plans/2026-06-10-render-perturbation-if-lines.md at vfm · Arthurshen926/ICLPose · GitHub"
[3]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/vfm/matcha_coarse_to_fine.py "ICLPose/feature_extract/vfm/matcha_coarse_to_fine.py at vfm · Arthurshen926/ICLPose · GitHub"
[4]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py "ICLPose/feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py at vfm · Arthurshen926/ICLPose · GitHub"
[5]: https://github.com/Arthurshen926/ICLPose/blob/vfm/docs/superpowers/plans/2026-06-09-perturbation-aware-geometric-adapter.md "ICLPose/docs/superpowers/plans/2026-06-09-perturbation-aware-geometric-adapter.md at vfm · Arthurshen926/ICLPose · GitHub"
[6]: https://github.com/Arthurshen926/ICLPose/commits/vfm/ "Commits · Arthurshen926/ICLPose · GitHub"
[7]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/tools/vfm/diagnose_render_pose_perturbation.py "ICLPose/feature_extract/tools/vfm/diagnose_render_pose_perturbation.py at vfm · Arthurshen926/ICLPose · GitHub"
