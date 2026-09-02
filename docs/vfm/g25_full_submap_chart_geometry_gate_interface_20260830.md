# G25 full-submap chart geometry gate：独立接口与当前 readiness

> **Superseded topology note:** the v2 topology described below omitted
> source-reference physical edge safety and is diagnostic-only.  The current
> v3 contract, real artifacts and evaluated result are recorded in
> `docs/vfm/g25_reference_safe_v3_explicit_atlas_gate_20260830.md`.

> dense-v4 输入闭包与最新真实 readiness 见
> `docs/vfm/g25_strict_full_submap_gate_inputs_20260830.md`。下文早期
> readiness 记录不应覆盖 sealed-v2 / equal-budget M0 新合同。

## 结论边界

这个 gate 只回答一个问题：在完全相同的 held mapping rays、相同物理子地图边界和冻结 chart inventory 上，MAtCha explicit chart atlas（M1=DAV2，M2=MoGe-3）相对一个**同 source evidence budget**的 M0 表现如何。当前可运行 M0 是 source-reference surface control，不是 2DGS；full-train 2DGS 只能作为 unequal-budget diagnostic。

它不读取 Cambridge query 图像、query pose 或 query GT，不训练、不修改 selector/alignment，也不把 held MASt3R pointmap 冒充传感器深度真值。held reference 只是 source-disjoint mapping-geometry diagnostic。

## 强制输入

- physically isolated `goal_maplet_disjoint_chart_upstream_authority_v2`；
- `goal_maplet_strict_held_ray_inventory_v1`，必须精确覆盖 authority 的全部 held ordered names，包含 dense depth、world normal、boundary 与 block ID；
- M0 必须是同 source ordered pool 的 bounded control；当前是 `goal_maplet_source_only_bounded_surface_baseline_v1`；
- M1/M2 各一份 pre-alignment atlas 和一份 post-alignment atlas；
- sealed `goal_maplet_chart_comparison_domain_v2`，必须显式冻结 exact sampled pixel/face indices，并绑定 v2 authority 与 frozen plan；
- M1/M2 必须共享逐字相同的 chart names、vertex/face offsets、faces 和 UV；
- model-neutral source-only frozen `ChartSubmapPlan`；seam 只审计它已经冻结的 coverage edges。

authority content hash、comparison-domain content hash、bounded-submap content hash 或 bounding box 缺失/不一致时，程序输出 `FAIL_CLOSED / NOT_EVALUATED / KILL_INPUT_CONTRACT`，不产生性能结论。
仅在多个文件中重复写一个相同字符串不算 lineage；evaluator 会重放 comparison-domain 文件 hash/content hash、plan content hash，以及 AABB 的 canonical bounded-submap hash。

M0/M1/M2 还必须声明并 replay 同一 source ordered image pool；`held_mapping_images_consumed=false` 且 `outside_frozen_source_mapping_images_consumed=false`。把 full-train 2DGS M0（已经看过 held mapping views）直接和 source-only chart atlas 比，会被 fail-closed，而不是被包装成 representation 优劣。

## 指标与判定

主指标是 macro-view good-ray recall：

```text
good = reference_valid
       AND rendered
       AND |z_render-z_ref| <= max(0.50 m, 0.05*z_ref)
recall = sum(good) / sum(reference_valid)
```

因此未渲染 ray 必须失败。仅在成功渲染像素上计算的 AbsRel median/P90 是次指标，不允许替代 recall。

同时报告：

- missing=fail 的 normal recall@10/20/30 和 joint depth+normal recall；
- 无 chart-parameterization seam 的 physical boundary F1；
- per-view macro summary，不用 pooled pixel 让长/密视图支配结论；
- M1-M0、M2-M0、M2-M1 的 paired block bootstrap CI；
- initial→aligned 的 face area ratio、UV-to-3D Jacobian singular-ratio、face flip/collapse/expansion；先剥离一个允许恢复的全局 similarity scale，再判局部/各向异性形变，绝对 metric scale 仍由 held rays 判；
- 仅 frozen coverage edges 上的 symmetric seam thickness、point-to-plane、normal angle 和支持率。

machine-readable 决策分开给出：M1 vs M0、M2 vs M0、M2 vs M1 non-inferiority、M2 vs M1 strict dominance、至少一个 initializer 是否支持 chart-atlas，以及两 initializer 是否都支持 initializer-independent 结论。

## CLI

```bash
PYTHONPATH=. python feature_extract/tools/vfm/evaluate_goal_maplet_full_submap_chart_geometry_gate.py \
  --m0_bounded_map M0.npz \
  --m1_initial_atlas M1_initial.npz \
  --m1_atlas M1_aligned.npz \
  --m2_initial_atlas M2_initial.npz \
  --m2_atlas M2_aligned.npz \
  --held_ray_inventory held_rays.npz \
  --disjoint_upstream_authority authority_v2.json \
  --comparison_domain selected_common_domain.npz \
  --frozen_submap_plan source_only_plan.npz \
  --output full_submap_gate.json
```

输入不齐时 CLI 仍写出可解析的 fail-closed JSON，并以状态码 2 退出。

## 2026-08-30 真实输入 readiness

已经存在且可 replay：

- v2 physically isolated authority；
- 19-view source-only primary selector plan；
- all-19 comparison-domain diagnostic；
- DAV2/MoGe-3 source initializer inventories。

但真实 performance gate 目前不能运行：

- primary plan 只有 1 条 coverage/alignment edge，`selected=0`、`operational_submap_count=0`；
- all-19 comparison domain 被明确标为 diagnostic，不能代替一个通过 coverage gate 的 selected submap；
- 尚无 strict held-ray inventory；
- 尚无 paired M1/M2 initial + aligned full-submap atlas；
- 现有 M0 planar files 没有冻结 bounded-submap hash 和 bounding box。
- 现有 full-train M0 也没有证明与 source-only M1/M2 使用相同 mapping input pool；held seq1/seq9 很可能已经参与其 2DGS 建图，不能直接作为公平 representation baseline。

因此当前唯一科学上合法的机器结论是 `NOT_EVALUATED`，不能宣称 MAtCha chart atlas 优于或劣于 M0。第一阻塞点是得到一个 model-neutral、至少含可连通共视组件的 source-only operational submap；其后才能冻结共同边界、构造 held rays 和运行本 gate。

## 合成验证

`tests/test_goal_maplet_full_submap_chart_geometry_gate.py` 覆盖：完美 GO、selection-rank 顺序、缺失 ray KILL、各向异性塌缩 KILL、纯全局尺度不误判、comparison-domain mismatch fail-closed、非 frozen edge 不进入 seam、boundary tolerance。
