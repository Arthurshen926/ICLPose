# 2026-05-16 POFD Stage3a q50 Radius/Scale Verification

## Goal

最新专家建议把 q50 问题收窄为 score surface 排序不稳定，而不是 candidate 不够、selector 不够、confidence 不够。按这个判断，本轮只验证一个核心问题：

```text
q50 的正确匹配峰值是否落在当前 high-res small-radius scorer 的搜索窗口外？
```

## Implemented

- `train_nvs_pose_feature_adapter.py`
  - 新增 `score_feature_hw`，允许只在 candidate scoring 路径下采样到低分辨率，不改变 query/map 主特征监督分辨率。
  - 支持 `score_mode=pair_matcher_local` + larger radius 的主训练/eval 路径。
  - 新增 candidate selection bias metrics：identity fraction、selected/oracle delta、best-vs-identity score margin。
- `eval_feature_pose_audit.py`
  - 支持 adapter checkpoint、cache candidate bank、`pair_matcher_local` audit。
  - 解决早期 lattice audit 和 Stage2d cache 协议不一致的问题。
- `report_pofd_basin.py`
  - 报告 identity/near-init bias 和 score margin。
- 新配置：
  - `feature_extract/configs/pofd_stage3a_multiscale_score_q50_shuf.yaml`

## q50 Audit Result

Checkpoint：Stage2d canonical POFD adapter。候选：cache topK，与 Stage2d 训练/验证协议一致。

| score path | pred cost | top1 | spearman | conclusion |
|---|---:|---:|---:|---|
| 68x120 / r3 | 0.285m | 0.542 | 0.103 | 原 high-res small window 排序弱 |
| 68x120 / r8 | 0.258m | 0.625 | 0.297 | 半径增大后明显改善 |
| 68x120 / r12 | 0.236m | 0.708 | 0.374 | 排序继续改善 |
| 34x60 / r5 | 0.177m | 0.833 | 0.435 | 低分辨率 basin layer 有效 |
| 34x60 / r12 | 0.171m | 0.875 | 0.506 | 当前最佳 audit setting |

结论：专家关于 local search range 的判断成立。q50 不是单纯 feature health 问题，而是 high-res/small-radius 下正确 pose evidence 没被 scorer 看到。

## Stage2d vs Stage3a Formal Eval

两者使用同一 query/map checkpoint、同一 q50 cache candidate 协议。Stage3a 只改变 candidate score surface：`34x60 + pair_matcher_local radius=12`。

| run | pred cost | oracle | gap | top1 | spearman | succ@25cm/10deg | trans | rot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage2d current | 0.316m | 0.129m | 0.187m | 0.583 | 0.082 | 0.667 | 0.305m | 6.22deg |
| Stage3a eval-only | 0.198m | 0.129m | 0.069m | 0.833 | 0.486 | 0.833 | 0.193m | 3.02deg |
| Stage3a short train best | 0.173m | 0.129m | 0.043m | 0.875 | 0.518 | 0.875 | 0.168m | 2.66deg |

Stage3a 已达到专家建议 Gate 3 的主要排序门槛：

```text
pred_cost < 0.20m
top1 > 0.80
selected-oracle gap < 0.07m
spearman > 0.45
```

## Identity Bias Check

Stage2d / Stage3a 的 `selected_identity_frac` 都是 `0.0`，并且 Stage3a 的 `score_best_minus_score_identity` 明显为正。因此 q50 主失败不是 identity/near-init bias，而是 medium perturbation 下 local score surface 看不到或压不住正确峰值。

## Training Notes

- `34x60 + radius12` 反传显存压力大。
- `batch=8/4` 在 3090 上 OOM；稳定可跑组合是 `batch=2, --no-amp`。
- dense pair-match teacher loss 在 radius12 下过重，所以 Stage3a config 暂时设 `pair_matcher_weight: 0.0`。rank/teacher-quality/observability losses 仍通过 pair matcher score surface 训练。
- `candidate_observability_margin_loss` 已经是 online score-hard-negative 形式：positive 是 topK 内 GT pose cost 最小候选，negative 是 cost 明显更差但当前 score 最高的候选。因此下一步不是新增 selector，而是继续围绕该 score surface 做 full validation 和更省显存的 multiscale training。

## Next

1. 扩大 Stage3a best 到更完整 validation 集，确认 q50 提升不是小样本偶然。
2. 做 `34x60/r8` vs `34x60/r12` 训练对照，找精度/显存速度平衡。
3. 如果 full validation 仍保持 q50 `<0.20m`，再进入 Stage3c：semi-dense GeoNCE/NVS correspondence peak supervision。
4. 在 Stage3b/3c 稳定前，不开放 map-side adaptation，不回到 fine selector、confidence、Fisher/logdet、强 teacher anchor。

## 2026-05-16 Follow-up: True q50 Val128

发现 Stage2d 继承配置里有 `dataset.max_val_samples: 32`。之前的 `--eval-max-samples 64/128` 实际仍被 dataset cap 卡在 32。新增 eval-only companion configs：

- `feature_extract/configs/pofd_stage2d_nvs_quality_pose_margin_q50_val128_eval.yaml`
- `feature_extract/configs/pofd_stage3a_multiscale_score_q50_val128_eval.yaml`

它们只解除 validation cap，并关闭不完整的 val teacher-correspondence cache，不改变训练 checkpoint。

### Val128 results

| run | pred cost | oracle gap | top1 | spearman | succ@25cm/10deg |
|---|---:|---:|---:|---:|---:|
| Stage2d baseline, original score | 0.362m | 0.233m | 0.379 | 0.113 | 0.462 |
| Stage2d checkpoint + Stage3a r12 score | 0.244m | 0.114m | 0.688 | 0.516 | 0.703 |
| Stage3a best r8 | 0.240m | 0.111m | 0.680 | 0.496 | 0.695 |
| Stage3a best r12 | 0.233m | 0.104m | 0.703 | 0.526 | 0.719 |
| Stage3a best r16 eval-only | 0.228m | 0.098m | 0.711 | 0.544 | 0.727 |
| Stage3a best 17x30/r6 | 0.291m | 0.162m | 0.508 | 0.407 | 0.555 |
| Stage3a r12 continue, 40 steps | 0.225m | 0.095m | 0.711 | 0.535 | 0.734 |
| Stage3a r16 offset-chunk train, best step20 | 0.224m | 0.094m | 0.727 | 0.549 | 0.750 |
| Stage3a r12-continue checkpoint + r16 scorer | 0.223m | 0.093m | 0.719 | 0.550 | 0.742 |

结论：

- Stage3a 在完整 val128 上仍显著有效：`0.362m -> 0.225m`，top1 `0.379 -> 0.711`，spearman `0.113 -> 0.535`。
- 但完整 val128 上还没有达到 `<0.20m`，说明 32-sample Gate 3 是 optimistic subset。
- 主要收益来自 score path 本身，而不是短训：Stage2d checkpoint 只换 Stage3a r12 scorer 已经到 `0.244m`。
- r16 eval-only 略好，但 r16 反传在 3090 上 OOM，即使用 candidate chunk=1、batch=2 也不稳。它是后续省显存实现目标，不是当前训练主配置。
- 新增 offset chunking 后，r16 可以在 `batch=1, candidate_chunk=1, offset_chunk=32` 下稳定训练；`batch=2` 仍然 OOM。r16 最好结果约 `0.223m`，相比 r12-continue 的 `0.225m` 只有边际提升。
- 17x30 更低分辨率明显退化，不能继续盲目降分辨率。
- 目前 q50 完整 val128 仍未达到 `<0.20m`。主要瓶颈已经从“搜索窗口不够”转为“剩余 hard wrong candidates 的 score ordering 仍不够稳”，继续只调 radius/scale 的预期收益很低。

### Stage3c flow-NCE probe

已实现最小 Stage3c 基础设施：

- `pair_matcher_local` scoring 下可以叠加 `local_flow_nce_weight`。
- 低分辨率 score grid 会同步缩放 intrinsics 并 resize candidate world-position map，避免几何投影错位。
- 新配置：`feature_extract/configs/pofd_stage3c_multiscale_flow_nce_q50_val128.yaml`
- 新增 hard candidate filtering：`local_flow_nce_candidate_mode=best_and_hard_negative`，只对 topK 内 GT cost 最小候选和当前 score-high wrong candidate 做 flow-NCE。

短训结果：

| run | pred cost | oracle gap | top1 | spearman | succ@25cm/10deg | flow loss |
|---|---:|---:|---:|---:|---:|---:|
| Stage3c flow w=0.10, 20 steps | 0.228m | 0.098m | 0.711 | 0.531 | 0.734 | 4.94 |
| Stage3c flow w=0.05, 20 steps | 0.229m | 0.099m | 0.711 | 0.531 | 0.734 | 4.94 |
| Stage3c hard-flow w=0.05, r12 score | 0.224m | 0.095m | 0.719 | 0.528 | 0.742 | 4.69 |
| Stage3c hard-flow w=0.05, r16 score | 0.223m | 0.093m | 0.727 | 0.547 | 0.750 | 4.67 |
| Stage3c hard-flow w=0.02, r12 score | 0.224m | 0.095m | 0.719 | 0.529 | 0.742 | 4.69 |

结论：

- uniform flow-NCE 可运行，但没有超过 Stage3a。
- hard-flow filtering 生效：`local_flow_candidate_selected_frac=0.125`，即 top16 中只监督 oracle + hard negative；`local_flow_hard_negative_active=1.0`。
- hard-flow 比 r12 continue 有约 1mm 边际提升，但 r16 score 下仍约 `0.2228m`，没有超过当前最好 `0.2227m`。
- 因此当前 raw-correlation flow-NCE 不能 promoted。原因很可能是它优化 raw local correlation，而主排序路径是 `PairConditionedLocalMatcher` 的 pair-matcher score surface；下一版 Stage3c 应直接在 pair-matcher offset logits 上做 hard GeoNCE，而不是继续调 raw flow 权重。

### Current main checkpoint

当前 val128 最好主结果：

```text
result/result/feature_extract/pofd_stage3a_val128_r12_continue_b2_noamp_s40_20260516/checkpoints/best.pth
with r16 score path:
pred_cost=0.2227m, top1=0.7188, spearman=0.5498

trained r16 offset-chunk checkpoint:
result/result/feature_extract/pofd_stage3a_val128_r16_offsetchunk32_b1_noamp_s40_20260516/checkpoints/best.pth
pred_cost=0.2237m, top1=0.7266, spearman=0.5487
```

### Updated next steps

1. 保持 POFD Stage3a/r12-continue checkpoint + r16 score path 作为当前主线上限参考。
2. 不再把更多精力放在 radius/scale 上；r16 已经验证为有效但接近平台。
3. Stage3c flow-NCE 不放弃，但下一版必须改为 pair-matcher-aware、visibility/occlusion filtered、只围绕 oracle/good positive 与当前 high-score wrong negative 做 GeoNCE；当前 uniform flow-NCE 和 raw-correlation hard-flow 都不能 promoted。
4. 不回到 selector/confidence/Fisher/强 teacher anchor/map-side 解冻。
