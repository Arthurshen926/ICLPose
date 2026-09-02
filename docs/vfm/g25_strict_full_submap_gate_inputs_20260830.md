# G25 strict full-submap gate inputs（dense v4）

## 已封闭的输入合同

full-submap 主表现在只接受 sealed `goal_maplet_chart_comparison_domain_v3` 物理安全 topology。loader 必须实际找到并重放 `v3 -> exact-topology v2 -> optimizer v1` 三代 artifact；仅提供 metadata hash 不够。v3 必须声明并验证 `source_reference_edge_safe=true`，v2 仅保留为 legacy diagnostic，不能直接进入正式 gate。

held-ray builder 分两阶段执行：

1. 只打开 isolated source seq4、frozen v4 plan、sealed v3 domain、DAV2/MoGe3 source initializers；在 v3 reference-safe face-referenced vertices 上取 source MASt3R world points、DAV2 world points、MoGe3 camera-to-world points的精确并集，再加固定 1 m margin，冻结 AABB；
2. AABB 设为不可变后才打开 authority 指定的 isolated held seq1 cameras/pointmaps。12 个 held view 全部按 authority order 写入，missing render 在评估时计失败。held MASt3R 只是 source-disjoint mapping diagnostic，不是 sensor depth GT。

任何 authority/plan/domain/tree/initializer/pointmap hash 不一致、任何 held view 在 source-only AABB 中没有可靠 ray、或者 phase-order metadata 缺失，均 fail-closed。

## 公平 M0

当前可运行 M0 是 `goal_maplet_source_only_bounded_surface_baseline_v1`：同一 isolated source ordered pool 的 MASt3R reference surface，且使用和 M1/M2 完全相同的 v3 physical-safe face topology。它的合法科学结论只限于：

```text
chart atlas vs equal-budget source-reference surface control
```

它不是 2DGS，不能被写成“chart atlas vs 2DGS”。真正的 2DGS 主表基线需要另行从 exact authority source inventory 训练并封存训练 manifest。现有 full-train 2DGS 看过 source pool 之外的 mapping views，只允许标为 `unequal_budget_diagnostic_only`。

## 真实 CPU smoke

旧的 3-chart v2 CPU-only smoke 已成功，但现在只是 legacy negative control：

- selected order：`seq4__frame00275.png`、`seq4__frame00286.png`、`seq4__frame00282.png`；
- bounded hash：`855faa21324cd4c1c079e9e4bade0a7574699f1b81e4d595548a032e721f2271`；
- AABB min：`[-1.83195, -38.58570, -3.67497]`；max：`[19.11999, -3.14406, 22.00268]`；
- held view：12/12 非空，共 31,299 条 reference-valid ray；单 view 1,012–4,976；
- 无 GPU、无 Gaussian 训练、无 query image/pose/GT。

旧结果只证明输入构建合同和真实数据兼容，不是正式 v3 M0/M1/M2 性能结论。正式 v3 输入必须由下面命令重新构建，不得复用 v2-bound held/M0/bounds。

## 永久 artifact 构建命令

```bash
PYTHONPATH=. python feature_extract/tools/vfm/build_goal_maplet_strict_full_submap_gate_inputs.py \
  --disjoint_upstream_authority output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_isolated/disjoint_upstream_authority_v2.json \
  --comparison_domain_v3 output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/comparison_domain_operational3_reference_safe_exact_topology_v3.npz \
  --frozen_submap_plan output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_isolated/source_seq4_dense_overlap_aware_submap_plan_v2.npz \
  --dav2_initializers output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/dav2_source_seq4_24 \
  --moge3_initializers output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/moge3_source_seq4_24 \
  --output_held_rays output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_full_gate_v3/strict_held_seq1_rays_v1.npz \
  --output_source_surface_m0 output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_full_gate_v3/source_only_reference_surface_m0_v1.npz \
  --output_bounded_submap output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_full_gate_v3/bounded_submap_authority_v1.json \
  --output_audit output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_full_gate_v3/input_build_audit_v1.json
```

## 剩余阻塞

alignment runner 合法地继续消费 v1 domain（content `26d2ccf9…`）。四份 full-gate atlas export 必须同时做到：验证 alignment manifest 的 upstream v1 hash，然后按 sealed v3（content `3e6acde0…`）的 physical-safe exact indices 裁剪并绑定 v3。M1/M2 的 initial/aligned 四份 atlas 必须逐字共享 chart names、vertex/face offsets、faces、UV 和 bounded lineage，才可运行性能 gate。

报告判定固定拆分为四层：`relative_noninferiority` 只看 paired/block-bootstrap CI；`absolute_held_coverage` 要求 macro good-ray >= 20%、joint normal@20 >= 10%；`geometry_integrity` 独立审计；`composite_bounded_gate` 取三者合取。conditional AbsRel 与 boundary F1 仅诊断。即使相对 non-inferiority 为 GO，稀疏三臂也必须得到 absolute KILL、composite KILL；任何 bounded 结果的 `production_eligible` 永远为 false。

合成回归：

```bash
PYTHONPATH=. pytest -q \
  tests/test_goal_maplet_strict_full_submap_gate_inputs.py \
  tests/test_goal_maplet_full_submap_chart_geometry_gate.py \
  tests/test_goal_maplet_chart_comparison_exact_topology.py
```

覆盖 source-first/held-second 正例、held bytes 篡改、full-train 2DGS unequal-budget、exact source-only 2DGS classification、missing-ray、selection-rank、distortion、seam、identical-sparse 反例、v2 正式输入拒绝以及 v3 -> v2 -> v1 实体链重放。
