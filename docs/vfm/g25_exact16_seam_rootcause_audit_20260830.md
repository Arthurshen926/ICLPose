# G25 fresh exact16 seam 根因审计（2026-08-30）

## 结论

formal `full_submap_geometry_gate_exact16_v1` 的 `KILL` 结果没有被改写；本审计也没有修改共享 gate。但其中的 seam 子门不能被解释为“两种 initializer 都产生了整体错位的 chart atlas”。主要失败源是：

1. 当前 gate 在每条 coverage edge 上对两张 chart 的**全部稀疏顶点**做双向最近邻；它没有使用冻结的重叠对应，也没有计算点到目标三角形的距离。
2. 它把 `distance <= 1 m` 的所有最近邻都并入分位数，因此把“物理上靠近但不对应”的表面以及切向采样误差一起统计；条件 p90 自然堆积在 1 m 截断附近。
3. `p50 <= 0.1 m`、`p90 <= 0.3 m` 小于 stride-4 atlas 本身的采样间距。完全同拓扑的 source-reference M0 也在 48 条边上 **p50 0/48、p90 0/48**。
4. `coverage_edges` 表示两台 source camera 在 dense reference 上看见共同表面，不等价于最终稀疏 mesh 中存在可统计的 seam。48 条 coverage edge 中只有 42 条在 v3 输出 mesh 上存在至少一个 source-vertex 到 target-face 的冻结重心对应。

因此，当前“两臂 48 条边 p50/p90 全失败”主要是一个**必然 false-negative 的 seam metric**。与此同时，真实的局部几何问题仍然存在，最明确的是 frame 272；不能据此把整套几何直接放行。

## 审计边界

- correspondence、edge support、M0/M1/M2 几何统计全部只使用 frozen source plan、v3 common topology、source MASt3R M0、source-only alignment 输出。
- 没有打开 held pointmap、query 图像、query pose 或 GT。
- 读取现有 final report 只为逐字复核其 `overlap_seams` 字段；没有使用 held performance 字段来选择边、设阈值或生成 correspondence。
- 没有修改 `full_submap_chart_geometry_gate.py` 或任何共享 gate 代码。

## 当前 all-vertex NN metric 的反证

当前 metric 的精确定义是：对每条 frozen coverage edge，分别将 A 的全部导出顶点查询到 B 顶点 KD-tree、B 查询到 A；保留距离不超过 1 m 的结果，双向拼接后计算 p50/p90。support 下限为 `max(0.05, 0.5 * plan_overlap)`。

| 表征 | 有有限统计的边 | support 通过 | per-edge p50 中位数 | per-edge p90 中位数 | p50≤0.1 | p90≤0.3 |
|---|---:|---:|---:|---:|---:|---:|
| M0 source reference | 47/48 | 47/48 | 0.407 m | 0.866 m | 0/48 | 0/48 |
| M1 DAV2 aligned | 47/48 | 47/48 | 0.403 m | 0.863 m | 0/48 | 0/48 |
| M2 MoGe3 aligned | 47/48 | 47/48 | 0.396 m | 0.867 m | 0/48 | 0/48 |

M0 的单-chart 最近邻采样间距 p50 跨 chart 的中位数为 0.290 m，p90 中位数为 0.596 m；M1/M2 分别约为 0.285/0.594 m 和 0.293/0.600 m。也就是说，0.1/0.3 m 阈值低于表示分辨率。

更强的证据是 current metric 的 per-edge 排序几乎由 M0 拓扑决定：

- M0 与 M1 的 seam p50 Spearman 相关系数为 0.983，p90 为 0.850。
- M0 与 M2 的 seam p50 Spearman 相关系数为 0.970，p90 为 0.909。
- M1/M2 current Euclidean p90 约 0.844/0.846 m，而 point-to-plane p90 只有 0.261/0.231 m；逐边切向分量 p90 中位数约 0.822/0.831 m。

这说明 current p50/p90 首先量到的是稀疏采样和错误 correspondence，而非两臂输出的真实法向错位。

唯一没有有限 current-NN 统计的边是 279–283：chart 283 只有 12 个顶点、6 个面，两个导出顶点集之间没有距离不超过 1 m 的点。dense reference 上该边仍有真实共同表面，所以这是“最终稀疏表示不支持该 seam metric”，不是 plan 中凭空产生的非重叠边。

## 两种 source-only 冻结 correspondence

### Dense target-pixel 诊断

该诊断不是最终建议的 gate，只用来隔离 target sparse-vertex NN 偏置：

1. source 是 v3 topology 中由面引用的 stride-4 顶点。
2. 使用 M0 world point 和 source camera 将其连续投影到 target 的 144×256 common-domain grid。
3. target 是投影坐标经 `numpy.rint` 得到的 dense target pixel，**不是**导出的 stride-4 target vertex，也不是三角形重心插值。
4. correspondence 仅由 M0 冻结：正深度、在图内、target common-valid、`|z_s-z_t| <= max(0.30 m, 0.025*near_depth)`、unsigned normal 不超过 35°、逐对应 same-camera-side。
5. 冻结的 pixel pair 原样用于 M0/M1/M2；M1/M2 dense point 来自 `charts_data.pts / scale_factor`。

48 条边都有统计，共 4197 个双向 correspondence。这里的“38/48”指 **48 条都有 Euclidean p90，其中 38 条满足 0.3 m**，不是只有 38 条有统计。

| 表征 | Euclidean p50≤0.1 | Euclidean p90≤0.3 | symmetric p2plane p90≤0.3 | normal p90≤30° |
|---|---:|---:|---:|---:|
| M0 | 29/48 | 38/48 | 46/48 | 47/48 |
| M1 | 32/48 | 38/48 | 48/48 | 48/48 |
| M2 | 34/48 | 38/48 | 48/48 | 48/48 |

这表明两臂 alignment 的真实 overlap seam 至少不劣于 source-reference control；但 `rint` pixel 仍包含离散化，因此不能作为最终 Euclidean 绝对门。

### Target face + barycentric 诊断

这是更接近最终修复门的定义：

1. source 仍是 v3 face-referenced stride-4 vertex。
2. 在 target v3 精确 UV topology 中寻找包含连续投影坐标的 target triangle。
3. 用其确定性的 UV barycentric weight 冻结 `(source vertex, target face, weights)`；M0/M1/M2 复用完全相同的 face 与 weights。
4. M0 冻结过滤直接对 target triangle 的 M0 barycentric 3D point/depth/normal 执行相同 0.30 m、2.5%、35°、same-side 规则。
5. seam 距离比较 source vertex 与 target triangle barycentric 3D point，不再比较 target 最近顶点。

结果有 42/48 条边可审计，共 1704 个 correspondence；6 条 coverage edge 在最终稀疏 mesh 中没有 target-face correspondence。

| 表征 | 有统计的边 | Euclidean p50≤0.1 | Euclidean p90≤0.3 | symmetric p2plane p90≤0.3 | normal p90≤30° |
|---|---:|---:|---:|---:|---:|
| M0 | 42/48 | 27/42 | 36/42 | 40/42 | 42/42 |
| M1 | 42/48 | 33/42 | 35/42 | 41/42 | 42/42 |
| M2 | 42/48 | 34/42 | 36/42 | 41/42 | 42/42 |

M1/M2 的 barycentric point-to-plane 唯一绝对失败均为 272–273，且该边只有 4+2=6 个冻结对应：M1 p90 为 0.644 m，M2 为 0.590 m。这与 frame 272 的独立 same-UV 局部异常一致。

M0 本身在部分边的 Euclidean/p2plane 也超过绝对阈值，说明 source-reference 的跨视图不一致性必须作为 control floor；不能要求预测 atlas 比其用于冻结 correspondence 的 mapping reference 更严格地达到零 seam。

## coverage edge 与 seam edge 的区别

dense common-domain 投影复核表明，原 48 条 plan coverage edge 全部仍满足至少 0.08 的 symmetric source-reference overlap；不存在“多数 frozen edge 实际不重叠”的证据。

但 barycentric frozen count 的逐边分布为：min 0、p25 6、median 18.5、p75 37.75、max 186；6 条边为 0。若要求每个方向至少一个对应，39 条边仍组成一个连通图；若每方向至少 5 个，只剩 32 条边、3 个 component，chart 288 和 283 已孤立；每方向至少 20 个时只剩 11 条边、9 个 component。

因此当前 exact16 sparse topology 不足以支持“48 条 coverage edge 全部作为硬 seam”的理论假设。最小修复不能是放宽 0.1/0.3 m；必须先建立独立的、source-only 冻结的 auditable seam graph，并在图不连通时触发 topology densification 或 source-only reselection。

## 整体尺度/姿态与真实局部形变

alignment 后 M1/M2 对 M0 的 same-UV 全点距离分别为：median 0.073/0.072 m，p90 0.210/0.220 m。逐 chart Sim(3) 对 M0 的中位数约为：scale 0.997/0.997、rotation 0.24°/0.22°、translation 0.075/0.084 m。因此没有“16 张 chart 普遍存在整体尺度或姿态错位”的证据。

真实局部异常如下：

- frame 272：M1/M2 same-UV d90 为 0.725/1.121 m，point-to-plane p90 为 0.541/0.888 m，normal p90 为 60.4°/59.0°。即使每 chart 再拟合一次 Sim(3)，残差 p90 仍为 0.603/0.858 m，所以不是一个整体 scale/pose 参数可以修复的误差。
- frame 294：same-UV d90 为 0.287/0.337 m，normal p90 为 38.4°/44.2°；问题比 272 弱，而且其 barycentric overlap seam normal 仍通过，说明异常更可能位于非重叠区域或局部面片。
- M1 的现有 distortion `chart_area_ratio_max=2.747` 来自 frame 292 相对单一全局尺度的 chart-level 面积变化；M2 的 max 为 1.172。该信号是真实的 initializer-to-aligned 非一致修正，但它与 48/48 seam p50/p90 false-negative 是两个独立问题。后续 distortion 门应把“每 chart 的必要 Sim(3) 尺度校正”与“移除 Sim(3) 后的局部 Jacobian 形变”分别报告。

## 对 family/RADIO carrier 的影响

frame 272 占 atlas 顶点 3.97%、面 3.00%、M0 面积 1.76%；frame 294 占顶点 9.90%、面 10.83%、面积 7.06%。二者合计占 13.87% 顶点、13.84% 面、8.82% 面积。

source-only plan redundancy 复核：

- 保留 all16 时，24 个 source view 中 95.83% 达到 best surface support ≥0.25。
- 单独排除 272 或 294 均仍为 91.67%，高于原 0.90 coverage rule。
- 同时排除二者下降到 87.5%，不能在不替换 chart 的情况下全部移除。

因此：RADIO/UV appearance carrier 不必因为 272/294 的局部几何异常整体删除；但 pose backend 不能无权重地使用这两张 chart 的 3D/normal。最安全的 source-only 处理是先 quarantine frame 272 的 geometry、保留其 RADIO appearance descriptor，随后从 8 张未选 source chart 中 source-only reselect 一个几何替代；frame 294 使用 per-face geometry confidence/mask，而非整 chart 删除。完成替换前，不应宣称 family carrier 已可直接进入位姿后端。

## 不使用 held 调参的最小修复门

1. **M0 sanity invariant**：任何绝对 seam metric 必须先在同拓扑 M0 control 上可满足；M0 若失败，该 metric 标记为 `INVALID_CONTROL`，不能把同一失败解释为 M1/M2 geometry KILL。
2. **冻结 target-face correspondence**：在打开 held 之前，用 source M0 + final v3 topology 冻结 `(source vertex, target face, barycentric weights)`。正式 gate 禁止 all-vertex NN 和 arm-specific correspondence。
3. **分开 coverage graph 与 seam graph**：coverage graph 继续服务 source view coverage；seam graph 只保留具有预先声明的双向最小 correspondence 数、目标 face support 和 M0 physical consistency 的边。统计所需最小数应按 p90 估计精度预注册，不能为当前结果临时设为 1 或 5。
4. **图结构 fail-close**：如果 auditable seam graph 不连通，不降低阈值；触发 stride/topology densification、保留更多 reference-safe faces，或在 24 张 source chart 内重新选择。选择和替换均不能读取 held。
5. **seam 主指标改为 control-relative**：以 target-face barycentric symmetric point-to-plane 和 normal 为主；Euclidean 作为辅指标。每条边报告 M1/M2 相对 M0 的 excess，而不是要求所有预测边低于一个连 M0 都不能满足的绝对值。
6. **单 chart 局部门**：在 source-only same-UV domain 上单独约束 aligned-to-M0 的 p2plane、normal、face validity，明确拦截 frame 272/294；这与 seam graph 门分开。
7. **形变分解**：先拟合每 chart Sim(3)，分别报告 chart scale/pose correction 与 Sim(3)-removed local Jacobian distortion。否则会把必要的 monocular scale correction 与局部折叠混为一谈。
8. **信息屏障保持不变**：上述 topology、edge、threshold、chart quarantine/reselection 全部封存后，才允许重新运行 fresh held coverage；held 只作最终验证，不能返回改变 seam graph。

在完成这一步之前，合理结论是：aligned chart geometry 已显示出可用的主体质量，当前 seam KILL 主要是门设计错误；但 sparse seam support 和 frame 272 局部异常仍使其不具备生产/位姿后端放行条件。

## 审计产物

- Machine-readable 主审计：`output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_exact16_fresh/rootcause_audit_v1/exact16_seam_rootcause_audit_v1.json`
  - file SHA256: `cfa544f561459755de98a9865444badfa08abe317fa401c7437081b4e68c6a63`
  - canonical content SHA256: `53339e04a64a2ff5669d5595b0ee40aee7a4d31b96c977811060db8ac9ce692b`
- Current all-vertex NN M0 counterfactual：`output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_exact16_fresh/rootcause_audit_v1/exact16_current_vertex_nn_counterfactual_v1.json`
  - file SHA256: `4cf0d75e92512aff386aad9ae63e940b03f7b1d5412d7ef2f94ad78a2e2027ad`
  - canonical content SHA256: `4d67e31abe7d5afc30c9934d8aff50d666fed48245f58ecb616b4baf7ba3cac9`
- 逐边 CSV：`output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_exact16_fresh/rootcause_audit_v1/exact16_seam_rootcause_per_edge_v1.csv`
  - file SHA256: `c8f1b89b4c2523e9ab3dce0314f90188ae67f8bd8eb07bd49c446d3c529a5f1f`
- 汇总图：`output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_exact16_fresh/rootcause_audit_v1/exact16_seam_rootcause_summary_v1.png`
  - file SHA256: `fb71b6ecc015b39da08d5588fa848a03e9ae4e3e820071735e3f9e4a3eb01e16`

