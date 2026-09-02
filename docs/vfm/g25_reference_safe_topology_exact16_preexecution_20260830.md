# G25 reference-safe topology / exact16 / fresh-held preexecution（2026-08-30）

## 结论

本轮完成三个彼此独立、均 fail-closed 的修复：

1. paired optimizer 仍使用 v1 common pixel domain，但正式 full gate 的 face topology
   升级为 source-only MASt3R reference edge-safe v3；
2. source-only selector 新增 pre-frozen minimum cardinality，exact16 只有选满 16 且
   coverage 达标才 operational；
3. 从旧 pose-only eligible seq1 库存中排除已经用于 diagnostic 的 12 帧，在任何新
   RGB/pointmap 打开前冻结一组 disjoint fresh 12 帧和完整物理隔离命令合同。

没有放宽 depth、normal、baseline、triangulation、reference edge 或 held gate。

## Physical reference-safe topology v3

实现：

- `feature_extract/vfm/localization_goal_maplet/chart_comparison_reference_safe_domain.py`
- `feature_extract/tools/vfm/seal_goal_maplet_chart_comparison_reference_safe_v3.py`
- `tests/test_goal_maplet_chart_comparison_reference_safe_v3.py`

v3 face mask 定义：

```text
face_valid_v3 = face_valid_v2_DAV2_MoGe
                AND source_only_MASt3R_reference_edge_safe
```

对每个 upstream-valid stride patch 检查 inclusive patch 内所有水平、垂直 unit-pixel
world edge。阈值固定为：

```text
max(0.5 m, 0.05 * median(||point_world - camera_center_world||))
```

reference world points 来自严格 source root 的 MASt3R pointmap，并以
`cv2.INTER_AREA` resize 到 optimizer pixel domain。v3 重新 packed topology，只保留被
surviving face 引用的顶点。

真实 3-chart v3：

- file:
  `output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/comparison_domain_operational3_reference_safe_exact_topology_v3.npz`
- file SHA: `59c97f68279062b95783f689e20b14d777a25d70dccba59fb3f8b90a1e2ce80f`
- content SHA: `3e6acde0777afa58f287054c18b73aeafdfe1a1e5d0d45caf1304865d8b94656`
- stride 4: `1085 -> 1081` safe quads；最大被删 reference edge `6.4867 m`；
- stride 8: `156 -> 156`。

## exact16 source plan

selector 增加：

- `minimum_selected_charts_per_submap`；
- stop condition 必须同时满足 `selected >= minimum` 和 coverage target；
- component 小于 frozen minimum 直接 fail；
- v3 plan metadata 明示 held geometry 没有参与选择。

真实 plan：

- file:
  `output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_isolated/source_seq4_dense_exact16_overlap_aware_submap_plan_v3.npz`
- schema: `goal_maplet_overlap_aware_chart_submap_plan_v3`；
- file SHA: `b2f1dc42321c911cf4036b176c41a34bc51bafbedbe6cab0478b9a4b56a8cbff`；
- content SHA: `45c139ff518ddb58f2fc471bed768173991a9992f784cc4e6c86d0fc17999c1e`；
- exact order SHA: `ea8e7cdcd82569bcf4b1b4c81bcfc70a950ec0bfb4ace878a6170adcb096ee8e`；
- selected order:
  `275, 286, 282, 294, 272, 271, 292, 273, 290, 280, 281, 291, 288, 274, 289, 287`；
- base/strict/loose 均 selected=16、operational=1；base coverage=100%；
  strict/loose 与 base selected Jaccard 均为 `0.7778`。

## exact16 stride eligibility 修复

正式 full gate 预冻结只消费 stride 4。旧 sealer 错把未消费的 stride 8 也要求为每张
chart 非空，导致 exact16 v1 的五张 chart 被非原则性拒绝。

新 contract：

- `full_submap_gate_eligible_strides=[4]`；
- stride 4 每张 chart 必须至少一个 quad，否则拒绝；
- stride 8 可为空，但逐 stride / chart 的 quad、triangle、packed vertex counts 和 empty
  names 必须进入 metadata；
- exporter/full gate 只能消费 frozen eligible stride 4，不能自行筛选或退回 stride 8；
- 没有修改任何几何阈值。

exact16 v2：

- file:
  `output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_exact16_authority/comparison_domain_exact16_exact_topology_v2.npz`
- file SHA: `835f790a2821d7e2994c8bb5fae4a5a26b67772969851768932fe3e8d9a87c9b`；
- content SHA: `c8c160c4b7ab6644a34d08f6cf25fd4d4c09342f29ad42420724694b556ba282`；
- topology SHA: `2ec69b909f897366b96d4e4864dbfae0a9f3a691c14720e7c79c7e8ddcb4f4f0`。

exact16 physical v3：

- file:
  `output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_exact16_authority/comparison_domain_exact16_reference_safe_exact_topology_v3.npz`
- file SHA: `d67c06708b9603e9c61883f2a5a494ef27fd62cae2ae130ed61e14fdab7a86a1`；
- content SHA: `fb27455cf77c003dc771bb75fb5350c347adb31f07830bba8c7c0214255e257c`；
- arrays SHA: `5062825694dc1c1d9e0bd664f826c082dc67f955ef09a1ad4731d40d8ab3ef59`；
- topology SHA: `f0b53f2d67308f757e54331a8530cc4936976dae1a0a73b92e1de7f7665237b9`；
- stride 4 reference safety: `2111 -> 2031`，删 80；16 张均非空，最少 2 quads；
- stride 8: `266 -> 249`，五张为空并显式记录；它不是 gate-eligible stride。

## Fresh held complement preexecution seal

正式文件：

- `output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_preexecution/fresh_complement_held_physical_preexecution_plan_v1.json`
- file SHA: `b75105ce206d90eb0de8cc65cbe35e3d6b628709b6955ae2dba415dd6016c158`；
- content SHA: `16080bc7bcf6bddee0ea6e38a3017484216c04ec79b48613cf00e874841ea55a`。

它只读取旧 pose-only v2 JSON。seq1 eligible 30 中移除旧 diagnostic 12 后剩 18，按冻结
eligible 顺序执行 endpoint-inclusive：

```text
round(linspace(0, 17, 12)) = [0,2,3,5,6,8,9,11,12,14,15,17]
```

得到 global indices：

```text
[506,509,511,514,515,519,520,524,525,528,530,533]
```

对应 names：

```text
seq1 frames [173,176,178,181,182,186,187,191,192,195,197,200]
```

该集合与旧 12 完全 disjoint；选择发生在新 held RGB/pointmap 打开之前。JSON 内已冻结
physical-isolation input builder、SfM preexecution-contract builder 的完整 argv 和严格执行
顺序。目前未执行这些命令。

## 正式 full16 rebuild 接线

当前 exact16 v1/v2/v3 绑定旧 dense-v4 authority，只证明 source-side 构建与 topology
contract 可以通过。fresh-held 正式 gate 不得把它们与新 authority 混用。正确顺序为：

1. 执行 fresh complement JSON 中的 physical isolation builder；
2. 在任何 MASt3R role 启动前执行 SfM preexecution-contract builder；
3. 只运行 contract 发出的 exact source/held commands；
4. audit 新的 disjoint authority v2；
5. 只用新 source root 重建 exact16 plan v3，仍要求 min=max=16；
6. 重建绑定新 authority 的 DAV2 initializer inventory 和 optimizer v1 common domain；
7. 用新 plan/authority/source tree 从 v1 seal exact topology v2；
8. 用新 source pointmaps从 v2 seal physical reference-safe v3；
9. paired alignment runner 仍消费 v1；alignment 无需消费 v2/v3；
10. exporter/full gate 只消费 v3 的 exact packed stride-4 topology；held builder 绑定同一新
    authority，并拒绝旧 v2 或 stride-8 fallback。

每一层都必须 pin expected content SHA；不得只相信 internally self-consistent 文件。

## 验证与复杂度

对抗测试覆盖：hash substitution、source tree/pointmap mutation、v2/v1 substitution、source
reference discontinuity、camera-range relative threshold、orphan removal、minimum cardinality、
不足 16 component、fresh-held 重用、空 stride-4 拒绝和空 diagnostic stride-8 接受。

selector 是现有 pairwise source-only CPU 计算；cardinality change 不改变渐进复杂度。v3
reference safety每个 stride 为 `O(F * stride^2)` 时间，topology repack 为
`O(C*Gh*Gw + F)`，存储为 `O(V_used + F)`；source tree replay 为 `O(source bytes)`。
fresh-held complement 为 `O(E)`，只读一个 JSON。
