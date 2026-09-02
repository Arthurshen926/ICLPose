# G25 显式 chart atlas 几何门禁（2026-08-29）

## 结论

本轮支持继续探索“MoGe-3 初始化 + 多视图 chart alignment + chart-only deployment”，但不支持现在就替换 2DGS 主地图。

- `GO`：显式 chart/mesh 是更合适的部署载体；MoGe-3 在相同 8-chart、相同 1000 步对齐条件下改善跨-chart几何，但未在所有 held 指标上支配 DAV2。
- `GO`：当前 RADIO 全局召回可以保留，后续 carrier 从 primitive/child 改为 chart UV/region/submap。
- `KILL`：MoGe-3 单帧 point map 直接拼接不能作为最终地图。
- `KILL`：当前 8-chart gate 不能被表述为全场景 atlas、定位提升或 2DGS 已可退出。
- `KILL`：在 chart family、RADIO UV field、query-chart correspondence 和 pose oracle 完成前，不运行在线位姿优化。

本轮没有运行 Gaussian refinement，也没有训练/重建 2DGS。

## 修复的实际问题

### 1. MoGe-3 焦距画布错误

`cameras.json` 的 focal 定义在 512×288 chart canvas 上。旧初始化器把这个 focal 直接用于 1024×576 原图，导致水平 FOV 约放大一倍。现在强制：

```text
source_focal = camera_focal * source_width / camera_focal_canvas_width
```

新的 initializer schema 为 `goal_maplet_moge3_chart_initializer_v2`，必须显式给出 `--camera_focal_canvas_width 512`。旧 v1 initializer 和由它导出的 direct atlas 作废。

### 2. chart 行与 camera 行错位

MAtCha chart 数组按展平文件名词典序保存，而 `cameras.json` 不是同一顺序。现在 loader 用文件名建立双射，并以 chart depth 重投影逐 chart 重放验证；错序直接失败。

### 3. MoGe 无效区域进入 MAtCha 损失

原实现的 mask 只乘在 depth loss 上，matching、normal、curvature 仍会消费天空/孔洞的填充值；chart 构造还写死 `masks=None`。本轮修为：

- depth loss 只在 valid pixel 上归一化；
- normal 使用中心与四邻域均有效的腐蚀 mask；
- curvature 再按其法向邻域腐蚀；
- matcher 同时排除无效 source pixel 与投影到无效 target pixel 的匹配；
- matching denominator 改为有效 source domain × target chart 数；
- 显式 atlas 导出时保留 source valid mask，填充值绝不输出成 surface。

### 4. 跨 chart nearest 指标不精确

旧实现只查询全局前 32 个邻居；同 chart 稠密采样可能占满前 32，导致跨 chart 最近邻被漏掉。现在逐 chart 对“所有其他 chart”建树，得到 exact cross-chart nearest，并只在两端都有合法 mesh normal 时报告 point-to-plane/normal。

## 产物

### MoGe-3 pose-free 初始化

```text
/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/
  moge3_vitl_map_chart_initializers_focal_correct_v2/
```

- 46 个 map-route chart keyframes；
- 不读取 camera pose、query 或 GT；
- MoGe-3 ViT-L，resolution level 9，refine 3，FP32；
- manifest content SHA：`09c747c2527613152fdfb3cb15c8ca77ad1dc932521228126710b5a1632746df`；
- peak CUDA allocated：2.84 GB。

### 直接放置 negative control

```text
/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/
  moge3_direct_map_routes_focal_correct_control_stride4_v2.npz
```

- 46 charts，78,490 vertices，106,878 faces，3.90 MiB；
- exact cross-chart nearest median/P90：0.546/1.930 m；
- ≤0.25m / ≤0.5m：17.88% / 46.10%；
- `multi_view_alignment_applied=false`，只作 negative control。

### 同条件 bounded M1/M2

固定同一 seq4 的 8 个 keyframes、同一 MASt3R reference、同一 valid mask、同一 loss 和 1000 iterations：

```text
DAV2:
/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/
  dav2_masked_alignment1000_seq4_8chart_gate_v1/

MoGe-3:
/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/
  moge3_masked_alignment1000_seq4_8chart_gate_v1/
```

| 指标 | DAV2 M1 | MoGe-3 M2 |
|---|---:|---:|
| alignment wall | 86.30 s | 85.78 s |
| peak CUDA allocated | 0.727 GB | 0.727 GB |
| compact atlas | 0.207 MiB | 0.200 MiB |
| aligned cross-chart nearest median | 1.629 m | **1.534 m** |
| vertices with another chart ≤0.5m | 18.12% | **19.63%** |
| vertices with another chart ≤1m | 37.66% | **39.86%** |
| overlap point-to-plane median | 0.221 m | **0.181 m** |
| overlap point-to-plane P90 | **0.591 m** | 0.644 m |
| unsigned normal median | 33.01° | **30.44°** |
| held mapping-view triangle coverage | 28.98% | **29.59%** |
| held mapping-view depth median | **1.05 m** | 1.18 m |
| held relative depth median | **4.22%** | 4.61% |

说明：held reference 是冻结的 MASt3R mapping pointmap，不是传感器深度 GT；renderer 是 CPU reference triangle z-buffer，采用透视正确深度插值。因此只能做 M1/M2 的同口径 mapping diagnostic，不能当 Cambridge 深度真值。

## 科学解释

### 已证明

1. 修正内参后，MoGe-3 确实是有竞争力的 chart initializer；此前“MoGe-3 direct 更差”的结论由焦距 bug 污染。
2. mask-safe multi-view alignment 能显著增加 MoGe-3 chart 的近距离重叠，并将 overlap point-to-plane 中位误差压到约 18 cm。
3. 8-chart 的显式几何只有约 0.2 MiB，部署表示有希望远小于约 50 MiB 的当前 physical 2DGS map；但尚未计 RADIO UV feature 和全场景 family atlas。
4. MoGe-3 在跨 chart overlap/normal 上优于 DAV2，DAV2 在 held triangle depth 中位误差上略优；MoGe-3 值得进入更大全 submap gate，但尚未支配 DAV2。

### 尚未证明

1. 1000 步后 cross-chart distance P90 仍很大，说明一部分 chart 仍不重叠或被低频形变拉远；当前 loss 并未完成 canonical atlas。
2. held triangle render coverage约 29%，仍不能证明完整 surface coverage；P90 深度误差约18m，存在遮挡/远端错误。
3. 尚无 chart family fusion；8 个 source charts 仍是 8 个 view charts。
4. 尚无 RADIO feature 写入 chart UV，也没有 query→chart soft correspondence。
5. 尚未跑 plane/chart correspondence oracle 或真实 pose，所以不能声称定位精度提高。
6. M3 Gaussian-assisted refinement未运行；只有当全场景 route-clean M2 的 held geometry gate 先通过后，M3才有比较价值。

## 下一轮严格顺序

1. 在 route-clean mapping inventory 上做 overlap-aware keyframe selection，而不是等距抽帧。
2. 将 8-chart gate 扩到一个完整 facade/submap，改用 triangle rasterizer报告 held coverage/depth/normal。
3. 构建 chart overlap graph，融合为 surface families；source view不作为在线候选 ID。
4. 将冻结 RADIO token以坐标正确方式投到 chart UV，先做 GT-submap 的 chart correspondence oracle。
5. 比较 M0 2DGS plane、M1 DAV2 chart、M2 MoGe-3 chart 的同一 pose oracle；只有 M2 不低于 M0 才迁移主地图。
6. 若 M2 几何仍被远端漂移限制，再做 M3 offline Gaussian-assisted refinement；最终只导出 chart/mesh，在线不加载 Gaussian。

可视化：

```text
/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/
  masked_alignment1000_seq4_8chart_dav2_vs_moge3.png
```
