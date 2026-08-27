# G25 Planar Side Map + Pixel Surface Oracle v2

## 裁决

第二轮强 oracle 达到预注册的 80% `1m/10°` 门槛，正式裁决为：

`GO_TO_REAL_QUERY_PLANE_RECOVERY_GATE`

这推翻的是上一轮“平面主后端已经整体 KILL”的过强外推，不推翻上一轮对 voxel child、primitive-center visible refit 和一次 mass-WLS 的负结论。当前仍不是 production GO，因为 query plane 的 region ID、逐像素米制深度和对应关系均来自 GT 2DGS surface oracle。

## P0 修复与 side map

- eligibility 重新强制 `normal_cosine_p10`；
- map merge 法向阈值固定为 5°、plane distance 为 3 cm；
- 使用真实凸多边形间距，不再用外接圆近似；
- 每次 union 前对完整 prospective component 重新拟合；失败则原子回滚，原区域不丢失；
- `scale1/scale2` 明确是 2DGS Gaussian 标准差。旧微片边界只是显式 1-sigma support convention，不再冒充物理半轴；
- 修复 visibility renderer 将 compact depth 用 global primitive row 索引导致越界/错深度的问题。

两条预注册 side-map operating point：

| Side map | Components | Accepted / rejected merges | Weighted surface fraction |
|---|---:|---:|---:|
| strict: RMS 3 cm / P95 5 cm / normal P10 10° | 3,285 | 25 / 1 | 8.83% |
| balanced: RMS 5 cm / P95 10 cm / normal P10 15° | 4,113 | 51 / 5 | 14.01% |

child 只作为 bounded micro-patch seed，不是最终 plane identity。低全图面积覆盖不等于低查询可用率：在 GT 可见 surface 下，两套 side map 的 88/88 查询都有至少四个可解平面。

## P1/P2 强 oracle

对每个 query：

1. 在 GT pose 下运行 full-scene signed 2DGS visibility；
2. 每个有效 pixel 取 dominant visible primitive；
3. 将真实相机 ray 与该 2DGS primitive plane 相交，得到逐像素 surface XYZ；
4. 使用 oracle side-map region ID 聚合 pixel mask；
5. 在 mask 内拟合 query plane，不再使用 primitive center 或完整 ellipse covariance 作为 query measurement；
6. 比较 mass-WLS、normal-diverse inverse-variance GLS、robust IRLS。

固定分辨率为 256×144，每个 plane 至少20 pixels，最多16 planes。D-opt-like selection 使用 frozen inverse variance 与 normal Gram log-det；IRLS 使用固定 Huber 规则，无 held 调参。

## seq10 全88查询结果

### Balanced side map

| Solver | Metric 1m/10° | Scale-aware 1m/10° | Rotation median | Translation median / P90 |
|---|---:|---:|---:|---:|
| mass-WLS | 87.50% | 76.14% | 1.009° | 0.345 / 1.065 m |
| diverse GLS | 100% | 96.59% | 0.297° | 0.102 / 0.319 m |
| diverse robust IRLS | 100% | 96.59% | 0.297° | 0.057 / 0.181 m |

### Strict side map

| Solver | Metric 1m/10° | Scale-aware 1m/10° | Rotation median | Translation median / P90 |
|---|---:|---:|---:|---:|
| mass-WLS | 94.32% | 86.36% | 0.641° | 0.228 / 0.770 m |
| diverse GLS | 96.59% | 93.18% | 0.240° | 0.082 / 0.321 m |
| diverse robust IRLS | 96.59% | 93.18% | 0.240° | 0.033 / 0.191 m |

Balanced diverse-IRLS 的 plane mismatch 与 translation error 相关系数为 0.607；normal condition 与 translation error 为 0.295。普通 mass selection 与 diverse selection 的显著差距证明：rank 满秩不够，plane subset geometry 和 uncertainty 必须进入合同。

Scale-aware 设计矩阵中位 condition 仍约130，明显比 metric normal system 的约2.9更病态。96.59% 虽过门，但真实单目尺度恢复仍是重点风险。

## 准确边界

- `GO`：专用 planar side map、pixel/mask query plane、condition-aware robust solver 值得进入真实 query plane recovery 实验。
- `GO`：平面不应只作为 rotation sidecar；强 oracle 已证明它具备完整 6DoF 潜力。
- `KILL`：继续使用 voxel child identity、primitive-center visible refit、visible-mass Top16 和一次普通 LS。
- `NOT YET TESTED`：RADIO-to-plane region matching、MoGe/ZeroPlane query depth、真实 RGB query plane mask、bounded polygon/depth joint refinement、held seq12/seq14 end-to-end。
- `NONPRODUCTION`：本轮使用 GT pose 生成 pixel surface depth与oracle region correspondence；它只回答“若 query plane measurement 足够好，地图与 solver 是否可行”。

下一门应固定当前 side map 与 solver，在不读 pose 的 query RGB/RADIO 上恢复 plane mask、normal、relative/metric offset，并分别报告 plane recall、parameter error和最终 pose；只有真实 query recovery 仍接近此 oracle，才进入 region matcher与bounded refinement。

正式报告：`output/g25_pose_transport/map_disjoint_seq12_seq14_backend_v4_coordinate_calibration_disjoint/planar_pixel_oracle_seq10_v2/report_source_bound_v4_final.json`。
