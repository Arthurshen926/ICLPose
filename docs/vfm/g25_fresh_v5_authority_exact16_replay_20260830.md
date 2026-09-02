# G25 fresh v5 disjoint authority / exact16 source replay（2026-08-30）

## 结论

fresh source/held MASt3R 两个 sealed run 均自然 `exit 0`。本轮没有启动 initializer、
alignment 或 held-driven selector。签发 authority 前独立重放了：

- fresh complement preexecution plan content；
- physically isolated input manifest 及每张原图/拷贝图/sparse 文件字节；
- source 24 / held 12 的 exact global indices、names、routes；
- preexecution source/held exact argv、全 `image_idx` 库存、code/checkpoint bytes；
- source/held 完整输出 tree 和 pointmap inventory；
- posed-COLMAP frozen PINHOLE K、camera pose；
- source/held/forbidden route 与 image 集合不相交。

## Fresh disjoint v2 authority

- path:
  `output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_isolated/disjoint_upstream_authority_v2.json`
- file SHA: `f5135b46264333218ac32f5f23a87a8cc4a1e40f1cd219b83e9d89be639c8791`
- content SHA: `34b9acecf4fa7472bf39697827783ba900d22216d58901107750a1bdb7792a82`
- source tree SHA: `dc8f1c8743e622a3065be1e50e83c5cb51bb1813396ac550717d10dd1f07dd2f`
- held tree SHA: `31a6b8d9c8ea33c2276db0112091c0f8f40b3440d94bf6ac05ea8d84dcf05f75`
- source count: 24；held count: 12；
- source max pose replay error: `1.07e-14`；
- held max pose replay error: `2.13e-14`；
- source/held focal replay error: exactly zero。

authority 显式绑定 complement plan file/content、isolated input file/content、preexecution
file/content、新 held 在 geometry 前冻结、两个 process exit code，以及完整 tree/pose/K replay。

## Fresh source-only exact16 plan

- path:
  `output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_isolated/source_seq4_fresh_exact16_overlap_aware_submap_plan_v3.npz`
- audit:
  `output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_isolated/source_seq4_fresh_exact16_overlap_aware_submap_plan_v3.json`
- file SHA: `9eda83f588299b21ecd03bb3b282bef3fbc928781ac7126e5e3404df71bab4f1`
- content SHA: `cd639963a640de6f4861fff56b74143497b954975276b72f73c4f17f787f7e99`
- arrays SHA: `0165dadfc4e315668016fc75dfda84e73a1d8c4c0d73a2e18e1c9a7be2929ed5`
- selected order SHA:
  `15e9758381c401f557d47aa594e98ef965e95b1f067ab259a6873781e2660c7c`
- operational submaps: 1；selected count: exactly 16；base coverage: 100%。

exact selection order：

```text
275, 286, 282, 294, 272, 271, 292, 273,
290, 291, 288, 279, 281, 274, 289, 283
```

fresh source bytes 重新加载并重新执行完整 source-only pairwise selector 后，所有 plan arrays
及未封装 metadata 与落盘 plan 完全相同。

## 与旧 dense-v4 exact16 的稳定性

- 共同 selected charts: `14/16`；set Jaccard `14/18 = 0.7778`；
- exact ordered prefix: 前 9 张完全一致；
- old-only: `frame00280`, `frame00287`；
- fresh-only: `frame00279`, `frame00283`；
- coverage edges: `76 -> 67`；alignment edges: `72 -> 63`；
- non-isolated coverage charts: `22 -> 21`；
- base supported-view fraction: `1.0 -> 1.0`；
- mean selected surface support: `0.8766 -> 0.8867`；
- strict/base selected Jaccard: `0.7778 -> 0.8824`；
- loose/base selected Jaccard: `0.7778 -> 0.7778`。

结论：MASt3R fresh run 的 pointmap/tree 字节并不等于旧 run，因此不能沿用旧 plan hash；但
exact16 set-level 和前部 greedy 顺序保持较高稳定，coverage 仍完整，strict perturbation
稳定性没有恶化。正式 downstream 必须绑定新 authority `34b9...` 和新 plan `cd639...`。

## Legacy 3-chart stride-authority re-seal

为了兼容只接受显式 `full_submap_gate_eligible_strides=[4]` 的最终 gate，旧 3-chart
diagnostic 另起新路径重封，未修改旧 artifact：

- new v2 file/content:
  `42c7e3a85d157cba0a505d01267aec2d1273b69cb20f9a05e92e72d10190876b` /
  `12c04d6a9df3e91b350008df5d8a8ab6eeb105ead654cfcc1012c750ddee978d`；
- new physical v3 path:
  `output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/comparison_domain_operational3_reference_safe_stride_authority_v3.npz`；
- v3 file/content:
  `6ab5a08d3a348dd7c643623772b91d7bef85afca1a9d5af754fc41f9225448a9` /
  `41189d64f38773fc713bf164e858fdfa719181664cfc396d75edda856a7c9177`。

它仍绑定旧 v4 authority，只能作为 legacy 3-chart diagnostic；不得与 fresh v5 formal
full16 lineage 混用。
