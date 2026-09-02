# G25 overlap-aware chart submap 与 canonical family 门禁（2026-08-30）

## 结论

本轮把“等距抽帧”替换成了可审计的物理 overlap 规划，并把 source
view-chart 与 runtime canonical surface family 明确分开。但当前数据给出的
结论是 **KILL 当前稀疏 19-view source inventory，先补密候选，再做完整
M1/M2 alignment**，不是放宽阈值强行得到子地图。

- 主比较 selector 只使用 source-only MASt3R mapping pointmap 与 mapping
  camera；DAV2/MoGe-3 都不参与定义 inventory，因此没有 M2-specific selection
  bias。
- source/held 上游由物理隔离的 v2 authority 绑定：source=`seq4` 19 views，held=`seq1+seq9`
  18 views，source/held image 与 route 均 disjoint，`seq12/seq14` forbidden，未读
  query/GT。selector 还会重放 source 完整输出树哈希；held 树只由 authority 绑定，
  selector 不打开它。
- MoGe-3-only selector 仅保留为 system control，绝不能用作 M1/M2 主比较。
- 当前 8-chart aligned atlas 的 family carrier 可以构造，但只有 23.87% patch
  area 得到多 parameterization 支持，因此 **KILL 当前 atlas 的 family
  canonicalization**。

## Overlap 定义

对 source chart `i` 的有效 mapping surface sample 投影到 source chart `j`，只有
同时满足以下条件才计入 directional surface support：

1. 投影在 target frustum 内且 target pixel 有效；
2. 深度满足
   `|z_i-z_j| <= max(0.30 m, 0.025*min(z_i,z_j))`；
3. unsigned normal angle 不超过 35°；
4. 两个 mapping camera 位于局部 tangent plane 同侧；
5. camera forward angle 不超过 70°。

无向 overlap 取两个方向 support 的最小值，不用单向 max。coverage edge 还要求
该值至少 8%。alignment edge 进一步要求：

- absolute baseline 在 `[0.5, 20] m`；
- `baseline / median_overlap_depth` 在 `[0.025, 0.70]`；
- median triangulation angle 在 `[1.5°, 60°]`。

这些约束用于阻止 2--3 m 平行面、超远相机、同位姿重复帧和反侧表面误连。

一个 component 只有在以下 operational rule 全部通过时才可进入 bounded
alignment：

- 至少 3 个 source views；
- selected charts 在 alignment graph 中连通；
- 至少 90% component views 的 surface 有至少 25% 被 selected chart 支持；
- 不超过 16 个 selected charts。

这只是 mapping-side 完整 submap proxy，不声称 semantic complete facade，也不代替
后续 held-view triangle coverage。

## 实测

### Model-neutral source selector（主口径）

输入：新 disjoint authority 的 source `seq4` 19-view MASt3R pointmaps。

| 阈值 | coverage edges | alignment edges | operational submaps | selected |
|---|---:|---:|---:|---:|
| strict | 0 | 0 | 0 | 0 |
| base | 1 | 1 | 0 | 0 |
| loose | 3 | 3 | 1 | 3 |

base 下唯一 edge 是 `frame00074`--`frame00092`。相邻候选相机通常相隔约
10--13 m；当前 19-view inventory 是重建输入的稀疏等距抽样，本身没有形成稳定、
连通的 complete-facade overlap graph。loose 才出现的 3-view component 对阈值不
稳定，不能晋级。base pairwise surface-overlap p90 只有 0.00835。

当前 exact ordered selection 是空列表；这不是让 runner 自行补帧的信号，而是明确
的 fail-closed KILL。failed component 的 attempted selection 只留在诊断字段，绝不
进入可消费的 `selected_chart_names_in_order`。

主产物：

```text
/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/disjoint_upstream_v3_isolated/
  source_seq4_overlap_aware_submap_plan_v2.npz
  source_seq4_overlap_aware_submap_plan_v2.json
```

authority content SHA：`2748ce57861c9f023651554b4aa6ad1f096072ed1673b1be51956608e0889778`；
source tree SHA：`88a1ba09094fe472892d5bc8c197bf7c26e6c60c27c95e335b5808b2e8d89121`；
plan content SHA：`ed03adf587000f6d18c901df1ac57f3ea2e4e07d9296b31a4b3e5fd06c3df711`；
ordered-selection SHA（空列表）：
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`。

### MoGe-3 selector（system control only）

严格同口径在 46-view 历史 inventory 上得到 15 coverage edges、14 alignment
edges、2 个 operational components；但 strict/base selected Jaccard 只有 0.182，
且它由 M2 initializer 自身定义，不能用于 DAV2/MoGe-3 对比。

```text
/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/
  moge3_overlap_selector_system_control_strict_v2.npz
  moge3_overlap_selector_system_control_strict_v2.json
```

### Canonical surface-family carrier

在已有 MoGe-3 8-chart/1000-step atlas 上先按 chart 内 35° normal discontinuity
切 patch，再以 cross-chart distance/normal/point-to-plane overlap 合并。合并时禁止
同一个 anonymous parameterization 在一个 family 内出现两次，避免 transitive
collision。

| 指标 | 结果 |
|---|---:|
| patches | 231 |
| families | 227 |
| multi-parameterization families | 4 |
| multi-parameterization patch fraction | 3.46% |
| online-supported patch area | 23.87% |
| gate | KILL（要求至少 50%） |

runtime carrier 不保存 source image name/path/ID；source name 只在独立 offline
lineage sidecar 中出现。当前 carrier 只 canonicalize identity 与 UV
parameterization membership，还没有把成员几何融合成 shared mesh。

```text
/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/
  moge3_masked_alignment1000_seq4_8chart_gate_v1/
    canonical_surface_family_carrier_v2.npz
    canonical_surface_family_carrier_v2.json
    canonical_surface_family_lineage_v2.json
```

carrier content SHA：`ea30820ef40b3999a727ecc5d15c9c733a921932d8e2bc082b9d3cc611fd476d`。

## 正确的下一步

1. 在相同 source/held-disjoint authority 下生成更密的 `seq4` source candidate
   pointmaps；目标是候选相邻 baseline/overlap 足以形成稳定 graph，而不是直接把
   所有 views 变成 runtime charts。
2. 重跑本 selector；只有 strict/base/loose 都形成近似一致的 component/selection
   才冻结 ordered selected names 与 plan hash。
3. alignment runner 必须调用 `load_model_neutral_alignment_selection(...)`，同时
   提供冻结在实验 authority 内的 expected plan content hash，并逐 submap 使用 exact
   ordered names；不得 route/count 重采样。当前 plan 因 operational submap 为零会
   被该接口拒绝。
4. DAV2 与 MoGe-3 必须使用完全相同 selected names、validity 与 face inventory。
   face edge 的 relative threshold 必须用 camera depth/range，不能使用依赖世界原点的
   `||X_world||`。
5. 通过 held mapping-view triangle depth/normal/coverage 后再构建 family carrier；
   online-supported patch area 至少达到 50% 才进入 shared-mesh fusion 与 RADIO UV
   gate。

## 下一轮 pose-only densification 冻结

为避免再次对全 `seq4` 等距抽 19 帧，本轮又增加了一个严格 source-only 的相机窗口
规划器。它只 materialize posed-COLMAP 中 `seq4/seq1/seq9` 的 camera center 与
forward axis，跳过全部 2D/3D observations；source 排名完成之前 held camera 不会
传给排名函数。

固定扫描 `seq4` 的全部 306 个 24-view 连续窗口，先要求 adjacent baseline
`<=1.5 m`、adjacent turn `<=12°`、path `>=15 m`，然后按
`(endpoint displacement/path, path, earliest start)` 做 source-only lexicographic
选择。冻结结果为：

- source：global lexical indices `828..851`，即 `seq4 frame271--294`；
- adjacent baseline min/median/max = `0.541/0.731/1.001 m`；
- adjacent turn max = `8.06°`；path = `16.57 m`；endpoint displacement =
  `16.30 m`；straightness = `0.98398`；
- 306 个候选中 305 个通过；唯一 reject 是从 frame279 开始的窗口，path 小于
  `15 m`。全部候选而不只是 winner 都保存在 plan 中。

source 冻结后才进行 camera-only held 诊断。在 `6 m/45°` gate 下，`seq1`
frames172--201 有 30 个 eligible cameras；均匀冻结 12 个独立 rays（global indices
`505,508,510,513,516,518,521,523,526,529,531,534`），camera proxy 覆盖 source
24/24。`seq9` eligible=0，因此没有为了凑 route/count 强塞进 held inventory。这仍
不是 shared-surface 证明；v4 重建必须回到上面的 model-neutral overlap gate。

```text
/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/
  disjoint_upstream_v4_pose_only_plan/pose_only_densification_plan_v2.json
```

plan content SHA：`458f8080a2bc1382ef97db364458a7f0109359d3ccd83bb9a0b3a2fc71b148b0`；
file SHA：`84dbe9454cd8d76b78470dacc207f8045dc8700388ff123085933a7197528285`。

## 复杂度

- selector：`O(N^2 * H*W/s^2)` projection work，pairwise storage `O(N^2)`；当前
  CPU 19-view base+strict+loose 含 hash/JSON 读取约 14 s。
- pose-only window planner：固定 window 长度下 `O(N*W + H*W)`，其中 `N`
  为 source route camera 数、`H` 为 held camera 总数；当前 329-view `seq4` 全扫描约
  1.5 s CPU，plan 约 234 KB。
- chart patch split：近似 `O(F)` mesh adjacency。
- cross-chart family candidates：最坏 `O(P^2*S*log S)`，`P` 为 patch 数，`S`
  为每 patch 至多 512 个 deterministic samples。8-chart 实测约 2 s CPU。

可运行命令：

```bash
PYTHONPATH=/root/ICLPose python -m \
  feature_extract.tools.vfm.build_goal_maplet_chart_submap_plan \
  --selection_source mast3r_disjoint_source \
  --source_root /root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/disjoint_upstream_v3_isolated/source_seq4_19_mast3r \
  --disjoint_authority /root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/disjoint_upstream_v3_isolated/disjoint_upstream_authority_v2.json \
  --output /tmp/source_seq4_overlap_plan.npz

PYTHONPATH=/root/ICLPose python -m \
  feature_extract.tools.vfm.plan_goal_maplet_pose_only_chart_densification \
  --posed_colmap /root/ICLPose/output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g25_coordinate_correct_ideal_pinhole_oof_v2_authority/fold0/posed_colmap \
  --output /tmp/pose_only_densification_plan_v2.json

PYTHONPATH=/root/ICLPose python -m \
  feature_extract.tools.vfm.build_goal_maplet_chart_surface_families \
  --atlas /root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/moge3_masked_alignment1000_seq4_8chart_gate_v1/explicit_atlas_stride4.npz \
  --output /tmp/surface_family_carrier.npz \
  --lineage_output /tmp/surface_family_lineage.json
```
