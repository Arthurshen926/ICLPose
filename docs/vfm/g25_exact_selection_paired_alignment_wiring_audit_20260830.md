# G25 exact selection / paired alignment 接线审计（2026-08-30）

## 结论

当前共享 runner 与 comparison-domain builder 已经具备 model-neutral v2 plan、
expected plan hash、v2 disjoint authority、source-tree 和共同 pixel/quad mask 的初步
fail-closed hook。但要满足“两个 arm 使用 exact same face indices”，还差最后一层
显式 topology seal；仅保存 `face_valid_stride{4,8}` 布尔格不够直接证明 downstream
没有重新编号或保留 orphan vertices。

本轮新增独立 v2 sealer，不改动并行封板中的 runner/builder：

- 输入必须显式 pin upstream comparison-domain、plan、v2 authority、source tree 四个
  content SHA；
- 用 `load_model_neutral_alignment_selection` 获取 exact ordered names，要求恰好一个
  non-empty operational component；
- 重放 v2 authority content 和 source 完整树，不打开 held root；
- 由共同 pixel mask 与 frozen face-valid quads 生成 packed
  `sampled_vertex_offsets/pixel_indices/face_offsets/faces`；
- 只保留至少被一个 frozen face 引用的 sampled vertex，不允许 orphan vertex 污染
  AABB、seam、family 或 RADIO nodes；
- 输出 `goal_maplet_chart_comparison_domain_v2`，并设置
  `full_submap_gate_eligible=true`、`exact_face_indices_frozen_for_both_arms=true`。

旧 19-chart plan 的 operational component 为零，仍会在 topology 生成前明确拒绝；
不允许回退 route/count 等距抽帧。随后 dense v4 source-only plan 得到一个三 chart
operational component，本轮已实际封出：

- output: `output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/comparison_domain_operational3_exact_topology_v2.npz`
- official selection-rank order: `frame00275 -> frame00286 -> frame00282`；
- file SHA: `02a689cdc876cb43a9d1766f4b3fb823d6e85cec8af88728178d0e66525e124f`；
- content SHA: `84ff5f39a111a918c60c3ebb239b1ce3095f28f9ec8f4693cc70f24a6a26d811`；
- exact topology arrays SHA:
  `e1765ecdb7165af557d676b676ae750e63f9241b32088ccdbcd2a1277590c883`；
- mapping-source ordered inventory SHA:
  `c82dae30571a93b28c2f4c144753783d8b25e794a9517eda641b85c838700fc9`。

mapping-source 顺序不信任旧 v1 domain 的可选自报字段；sealer 重放 v2 authority 的完整
`source.ordered_names`，要求与 plan 的完整 `chart_names` 和
`source_ordered_names_sha256` 一致，再把绑定写入 sealed v2。

## 独立实现

- `feature_extract/vfm/localization_goal_maplet/chart_comparison_domain.py`
- `feature_extract/tools/vfm/seal_goal_maplet_chart_comparison_exact_topology.py`
- `tests/test_goal_maplet_chart_comparison_exact_topology.py`

对抗测试覆盖：

1. plan exact order（故意使用非 lexical order）；
2. zero-operational plan；
3. upstream domain hash substitution；
4. source-tree byte mutation；
5. domain names 重排后重新计算所有自哈希；
6. face indices 篡改后连 topology hash 也同步重算；
7. valid 但没有 frozen face 引用的 orphan sampled pixels。

独立回归结果：16 passed（exact topology、submap/family、explicit-atlas CPU tests）。

真实三 chart topology 统计：stride 4 的 packed vertices 分别为
`623/596/377`、triangles 为 `886/846/438`；stride 8 分别为
`122/101/59` 和 `130/128/54`。所有数组与 metadata/content hash 均已重放通过。

## 当前无共享文件修改的接线

当前封板 runner 的 loader 只接受 v1 schema 和四个 base arrays，因此不能直接把 sealed
v2（content `84ff...`）作为 `--comparison_domain`。本次 paired alignment 应继续传上游
v1 domain（content `26d2ccf9...`）；sealed v2 作为 downstream full-gate topology
authority。held-ray builder/evaluator 必须核对：

1. v2 的 `upstream_comparison_domain_content_sha256` 等于两臂 alignment manifest 记录的
   v1 content；
2. 两臂与 v2 绑定同一个 plan content、authority content、source tree 和 official
   selection-rank order；
3. 两臂最终 atlas 仅使用 v2 exact face-referenced packed vertices/faces。

这样无需在本次 order-fix 重跑前改共享 runner。若以后要求 runner 直接消费 v2，才采用
下一节的最小 shared-loader diff。

## Runner 最小接线 diff（尚未修改共享文件）

1. `_load_strict_comparison_domain` 接受 v2 schema，并读取
   `topology_array_names()`；对 base arrays 与 exact topology 分别重放，再调用
   `validate_exact_topology_arrays(...)`。
2. 强制 `full_submap_gate_eligible is True`、
   `exact_pixel_mask_frozen_for_both_arms is True`、
   `exact_face_indices_frozen_for_both_arms is True`。
3. 两臂的 `charts_data.npz` 除共同 `valid` 外，逐项复制 exact topology arrays；manifest
   记录相同的 `exact_topology_arrays_sha256`。
4. paired 模式始终调用 `_strict_plan_selection`。删除 paired 路径中的
   `_selected_names(cameras, route, chart_count)` fallback；如需保留历史实验，应改成另一个
   显式命名的 `legacy_unpaired_diagnostic` CLI，不能静默触发。
5. CLI authority 除文件路径外再要求 expected authority content SHA 与 expected source-tree
   SHA，避免“换一份内部自洽 authority”也被接受。
6. DAV2/MoGe 两臂必须使用 sealed per-chart initializer inventories；paired 路径不接受旧
   bulk `dav2_charts` fallback。

## Atlas/export 最小接线 diff

strict comparison path 必须从 frozen faces 先标记 used quad corners，再建立 local packed
vertex index。不得先把所有 `grid_valid` pixels 都加入 vertices。输出 atlas 的
`chart_vertex_offsets`、UV raster order、`chart_face_offsets`、`faces` 必须逐数组等于
domain 对应 stride 的 frozen topology；任何 arm-specific 删除、翻转或重编号都 fail。

## 命令

真实 dense v4 artifact 的可重放命令：

```bash
PYTHONPATH=/root/ICLPose python -m \
  feature_extract.tools.vfm.seal_goal_maplet_chart_comparison_exact_topology \
  --comparison_domain output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/comparison_domain_operational3_v2.npz \
  --expected_comparison_domain_content_sha256 26d2ccf9030ffe5813d25b29bc849aa6ab36f876db4d80a545de470e498c1a0b \
  --frozen_submap_plan output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_isolated/source_seq4_dense_overlap_aware_submap_plan_v2.npz \
  --expected_plan_content_sha256 3c507e10a28f9e1edfd5255d959bad06dac601b700b1b20974a5e504a533948b \
  --disjoint_upstream_authority output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_isolated/disjoint_upstream_authority_v2.json \
  --expected_disjoint_authority_content_sha256 bd6f7d7bd238b8d29e36dd1311b29cb0cf80c8f11bb1d7deb4280040c9d4f207 \
  --source_root output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_isolated/source_seq4_frames271_294_mast3r \
  --expected_source_tree_sha256 1b339a724855323426be13afda3a3aff9966bf1432566dfb2308afa778fb5eed \
  --output output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/comparison_domain_operational3_exact_topology_v2.npz
```

复杂度：每个 stride 为 `O(C*Gh*Gw + F)` 时间、`O(V_used+F)` 存储；source-tree
byte replay 为 `O(total source bytes)`。不使用 GPU，不读取 held geometry/query/GT。
