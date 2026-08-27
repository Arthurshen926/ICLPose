# G25 平面区域位姿路线：第一性原理 oracle 审计

## 结论

PlanaReLoc 式“平面区域匹配后直接求位姿”不应成为当前 Cambridge / 2DGS 地图的主位姿后端。它可保留为朝向约束和多假设旁路，但在 query 侧平面恢复前，平面 offset 已经被更强的可见 2DGS 几何 oracle 证明不足以稳定恢复平移。

本轮没有使用 ALIKE、PnP、全局离散 pose lattice，也没有训练或重建 Gaussian。

## 实现

- 从 physical child 的 exact primitive membership 拟合带边界平面；单个 2DGS ellipse 使用其均匀椭圆面内协方差，避免仅用中心点导致法向任意。
- 比较三种地图区域：atomic child、仅 parent 内有界共面合并、允许跨 parent 的有界共面合并。
- 对每个 seq10 查询选择可见质量最高的 16 个 oracle 对应区域。
- `ideal_parameter` 直接使用地图平面经真值位姿变换后的 query 参数，只验证求解器与可观测性。
- `visible_subset_refit` 只用每个区域实际可见的 2DGS primitive 重新拟合 query 平面，再求旋转和平移；这是比未来单目平面恢复更有利的几何 oracle。
- 固定尺度平移要求 normal matrix rank 3；未知统一尺度的平移求解有四个未知量 `[C_x,C_y,C_z,s]`，严格要求设计矩阵 rank 4。

## seq10（88 queries）结果

| 地图平面方案 | 区域数 | eligible surface | normal rank-3 | scale rank-4 | rotation median | translation median | 1m/10° metric | 1m/10° scale-aware |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Atomic child | 10,813 | 64.57% | 100% | 100% | 7.98° | 4.95 m | 4.55% | 0% |
| Parent 内有界合并 | 7,334 | 55.87% | 100% | 100% | 3.83° | 3.56 m | 14.77% | 10.23% |
| 跨 parent 合并 | 6,471 | 45.86% | 100% | 100% | 5.23° | 5.35 m | 5.68% | 1.14% |

所有 `ideal_parameter` 对照均为 100% 1m/10°，说明公式、符号、对应关系以及数值求解正确。`visible_subset_refit` 的失败因此定位为平面参数稳定性，而不是线性系统欠秩。最佳 parent 内合并只有 13/88 达标，远低于预注册 80% 可行性门。

跨 parent 合并恶化结果，支持“不能把重复/相邻立面无界合并成巨型平面”的设计约束。parent 内合并提高了可见质量覆盖与旋转稳定性，但平面 offset 的小误差仍沿平面法向被放大为米级平移误差；加入未知尺度进一步恶化条件。

## 决策

- `KILL`: plane-only translation / full 6DoF 主后端。
- `KILL`: 在该 oracle 门失败后继续投入真实 query plane matcher、MoGe/ZeroPlane 或 learned correspondence，企图掩盖地图平面 offset 问题。
- `GO`: 平面法向作为 RADIO 区域召回后的 orientation sidecar、退化检测、multihypothesis 分组或后端正则项。
- `GO`: 主线继续围绕 RADIO 的 token-layout 区域证据，设计连续的 query-conditioned position proposal；全局 104,808×60 仅保留为 support oracle。

正式报告：`output/g25_pose_transport/map_disjoint_seq12_seq14_backend_v4_coordinate_calibration_disjoint/planar_pose_oracle_seq10_v1/report_source_bound_v2.json`。

参考方法：[PlanaReLoc, arXiv:2603.20818](https://arxiv.org/html/2603.20818)。该工作面向室内平面图定位，并明确把单目平面恢复、重复区域和大场景歧义列为限制；本轮没有把其室内经验直接外推到 Cambridge 室外场景。
