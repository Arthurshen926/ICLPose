# G25：从物理表面召回到无对应点位姿估计

## 结论先行

这轮不改变已经验证有效的 pure RADIO child retrieval，也不引入 ALIKE、PnP、
新的 SfM 或新的 3DGS 训练。使用的地图仍是冻结的 3DGS、physical
parent/child maplet 与 canonical RADIO field。

当前主要故障已从“地图表面召回不到”转成“正确表面集合没有稳定地变成视角和
连续 6DoF 位姿”。因此把系统拆成三个互不混淆的合同：

1. **surface retrieval**：返回物理面积受控、允许多解的 child 集合；
2. **pose-basin acquisition**：由该集合取得若干有物理差异的视角/位姿 basin；
3. **pose verification/refinement**：在每个 basin 内用冻结地图渲染 RADIO support，
   以无硬对应点的能量重排并连续优化。

召回指标不能冒充定位指标，mapping view 也不能冒充最终 pose。

## 已实现流程

```text
query RGB
  -> RADIO 36x64 token grid
  -> token-to-child soft evidence
  -> area-budgeted child set                 [已有并冻结]
  -> feature-free child visibility atlas     [新增]
  -> 0.5m/5deg diverse chart centres         [新增]
  -> full-geometry occlusion render
  -> latent-child support + canonical RADIO  [新增]
  -> soft pose verification / local SE(3) energy
```

新增的 visibility atlas 只保存：mapping pose、每个视角可见的 child 分布，以及
固定 4x4 layout 中的 child 分布。它不保存 mapping RGB、mapping RADIO descriptor、
SfM point 或 keypoint correspondence。mapping pose 的角色仅是冻结地图上的
visibility chart sample。

另实现了 9x16 canonical RADIO feature atlas 作为诊断。它完全来自 frozen
canonical field，而不是 reference-image retrieval；真实结果表明它目前没有超过纯
child set score，因此不进入默认前端。

## 530-query 真实结果

数据是 Cambridge St Mary's Church 官方 530-query test split。该集合已经被反复
用于开发诊断，所以以下数字是 **development-exposed diagnostic**，不是未触碰的
最终测试声明。

### Visibility atlas 的几何上限

atlas 含 1,487 个 frozen mapping visibility samples，12,906 个 physical children。

| 阈值 | atlas 全库 oracle |
|---|---:|
| 0.5m / 5deg | 1.70% |
| 1m / 10deg | 21.70% |
| 2m / 20deg | 76.98% |
| 2m / 45deg | 90.19% |

这直接证明 stored mapping centres 只能给 coarse acquisition，不能承担最终精确
位姿。尤其 0.5m/5deg 的离散中心上限只有 1.70%，连续 refinement 不是可选项。

### Pure child-set chart acquisition

固定 set-only score、0.5m/5deg physical NMS：

| 指标 | Top1 | Top10 | Top32 | Top64 |
|---|---:|---:|---:|---:|
| 2m / 20deg | 18.49% | 53.02% | 67.92% | 71.51% |
| 2m / 45deg | 19.81% | 54.53% | 69.43% | 75.47% |
| 1m / 10deg | 5.28% | 15.85% | 20.00% | 21.51% |

Top64 的 best-pool median 为 1.086m / 12.76deg；Top1 median 为
3.774m / 10.57deg。说明 child identity 有效，但 orientation/layout ranking 明显不足。

9x16 canonical feature 与 child score 等权融合的 Top64 2m/45deg 同为 75.28%，
没有可证增益；继续增加静态全局 cosine 不是当前正确优化方向。

## 无对应点 soft surface pose energy

对固定 pose `T`，renderer 在完整 3DGS 遮挡下返回每个 query token 所看到的 child
与 canonical RADIO code。child identity 不被硬匹配，而是对 query token 的
`p(child | RADIO)` 做边缘化：

```text
support(T) = sum_i w_i [2 p_i(rendered_child(T)) - 1] / sum_i w_i
radio(T)   = sum_i w_i cosine(q_i, canonical_i(T)) / sum_i w_i
energy(T)  = (1-alpha) support(T) + alpha radio(T)
```

其中 `sum_i w_i` 对所有 pose 固定；不可见或缺 feature 的 token 取 -1 floor。
因此证据消失不能靠缩小分母提高分数。该合同没有 2D-3D hard match、RANSAC 或 PnP。

### 局部可观测性审计

在三条不同路线各取一个 query，以 GT 为中心，仅用于放置
`+-0.5m × xyz` 和 `+-5deg × xyz` 的 12 个探针：

- 18/18 个轴的中心二阶差分均为正；
- GT 在 13 个 probe 中排名分别为 3、4、1；
- 因而目标在真值附近确实有 6DoF 信息，但存在小的平移偏置，不能直接宣称为
  已校准 likelihood 或最终 pose objective。

这项审计只验证局部形状，不是 global localization success。

### 召回候选的真实重排

又选择两个具有明确 coarse failure 的 query 做 bounded exact verification：

- `seq13/frame00001`：正确的 1.583m/6.55deg basin 从 child rank 6 提升为
  soft-energy rank 1；原 Top1 为 6.459m/16.04deg。
- `seq3/frame00001`：正确的 0.467m/6.41deg basin 在 child rank 4，soft energy
  仍把 2.830m/6.90deg 的原 rank 1 排在第一，未改善。

因此该机制已经实际展示过“从集合召回中找回正确位姿 basin”，但两例只有一例成功，
还不能称为稳定 estimator。失败例也说明仅依赖 canonical feature cosine 与 visible-child
mass 仍会偏好错误但外观/支持相似的区域；后续必须保留更完整的 token layout，并用
多 basin 生存而不是过早 Top1。

## 理论边界

目前完备的部分：

- child retrieval 是 set-valued、物理面积受控、允许多解；
- pose acquisition 与 localization success 分离；
- chart proposal 排名不接触 query GT；
- soft energy 使用固定 query mass 和 unknown floor，缺失不能通过分母消失获益；
- child 是 latent support，不强迫不可靠的一一 token correspondence。

仍未完备的部分：

- stored mapping atlas 的 2m/45deg oracle 只有 90.19%；
- chart Top64 仅拿回 75.47%，orientation/layout posterior 尚弱；
- soft energy 只做了三个 query 的局部审计和 bounded candidate verification，尚未形成
  530-query 连续优化结果；
- 当前 renderer 是逐 pose、full geometry、同步 CPU/GPU 的慢路径，不能经济地对
  32--64 modes 做密集 refinement；
- 530 test 已开发暴露，之后不能再用它调 alpha、步长或 learned weights。

## 下一步优先级

1. 冻结当前 child retrieval，不再以几个小数点作为主线。
2. 把 full-token layout 变成 pose-equivariant scoring：保留 36x64 token，而不是再做
   global/4x4/9x16 pooling；学习量必须在独立 validation 上完成。
3. 先做 8--16 survivor 的 batched renderer，并把 geometry 常驻 GPU；这是真实的
   工程瓶颈，不解决就无法验证连续 refinement。
4. 在每个 coarse chart 内做多尺度 SE(3) trust-region/coordinate search；所有输出保留
   多 basin，不允许先压成 Top1。
5. 分别报告：surface recall、basin acquisition、exact survivor recall、refined pose、null、
   p50/p95 latency。禁止把 region distance 或 atlas oracle 写成最终定位精度。

## 产物

- `output/g25_pose_handoff/visibility_atlas_child_layout4x4_v1/atlas.npz`
- `output/g25_pose_handoff/visibility_atlas_child_layout4x4_v1/s0_setonly_top64.json`
- `output/g25_pose_handoff/canonical_feature_atlas_9x16_v1/atlas.npz`
- `output/g25_pose_handoff/soft_pose_energy_observability_v1/`
- `output/g25_pose_handoff/soft_pose_verification_v1/`
