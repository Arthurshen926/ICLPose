# 2026-05-16 ChatGPT-ICLPose Verification

## Scope

This file tracks the latest `ChatGPT-ICLPose.md` recommendations against the
current POFD implementation.  The active claim is controlled pose refinement:

```text
query image + 3D feature field + approximate T0
-> one pose-observable localization feature
-> local candidate/energy ranking and refinement-basin validation
```

Full single-image relocalization is explicitly outside the current main table.

## Verified Items

| Expert item | Implementation / experiment | Result |
|---|---|---|
| Reframe from RADIO compression to POFD | `docs/superpowers/plans/2026-05-16-pofd-reset-mainline.md`; stage configs use one localization feature and freeze old coarse/fine claims | Accepted as mainline |
| Single localization feature before coarse/fine | `pofd_stage1/2/2b/2c/2d` configs; no new trainable topK selector in mainline | Accepted |
| Basin curve evidence | `feature_extract/tools/report_pofd_basin.py` and `docs/superpowers/plans/2026-05-16-pofd-basin-report.md` | Implemented |
| Lightweight GT-vs-wrong pose observability surrogate | `observability_contrast_loss`; q10/q25 Stage2 best `0.0678m`, oracle `0.0453m` | Effective for small basin only |
| Confidence/uncertainty branch | Stage2b q10/q25 `0.0897m`, q50 `0.3549m` | Negative in current form |
| Candidate-internal pose margin | `candidate_observability_margin_loss`; Stage2c q50 `0.3464m` | Slight q50 gain over pure confidence, not enough |
| NVS teacher-quality + pose margin | Stage2d q50 best `0.3160m`, oracle `0.1294m` | Best 2026-05-16 POFD q50, still below old NVS `0.3013m` |
| Stronger anchor / residual texture health probes | Stage2e anchor `0.3894m`, Stage2e residual `0.3356m` | Negative; stronger health constraints alone do not fix q50 |
| Pose Fisher/logdet diagnostic | `feature_extract/pose_observability.py`; eval-only q10/q25 and q50 | Implemented; raw/unit logdet is diagnostic only, not a reliable success objective |
| Oracle/deploy split risk | `pose_refine/evaluate_pipeline.py` now has explicit `--eval_mode`; deploy mode rejects oracle LoFTR retrieval | Implemented guard |
| Config protocol hygiene | fixed `train_nvs_pose_feature_adapter.py` so YAML `training.max_steps/eval_every/batch_size` is respected unless CLI overrides | Fixed after Stage2e exposed the bug |

## Key Numbers

Best clean small basin:

```text
pofd_stage2_poseobs_q10q25: pred_cost=0.0678m, oracle=0.0453m,
top1=0.775, succ@10cm/5deg=0.925, succ@25cm/10deg=1.0
```

Best current q50:

```text
pofd_stage2d_nvs_quality_pose_margin_q50: pred_cost=0.3160m,
oracle=0.1294m, top1=0.583, succ@25cm/10deg=0.667
```

Old q50 NVS teacher-quality reference remains better:

```text
nvs_q50_train256_teacher_paircorr: pred_cost=0.3013m, oracle=0.1294m
```

Fisher/logdet diagnostic, 12-sample eval-only:

| Run | pred_cost | align | spearman | query logdet | map logdet | conclusion |
|---|---:|---:|---:|---:|---:|---|
| q10/q25 Stage2 | `0.0611m` | `0.968` | `0.635` | `3.23` | `1.27` | good ranking and alignment |
| q50 Stage2d | `0.2799m` | `0.687` | `0.078` | `16.04` | `13.91` | high gradient/logdet but poor ordering |

This verifies an important negative result: **logdet/condition alone can be
inflated by feature scale or high-frequency variation and does not guarantee a
correct pose-energy basin.** It should be logged as a health/diagnostic metric,
not used alone as the core training objective.

## Current Conclusion

The expert reset was useful, but the remaining bottleneck is now sharper:

```text
q50 does not fail because there is no selector head.
q50 fails because the learned localization feature does not produce a
validation-stable query-map score surface for medium perturbations.
```

The most productive next architecture change is not stronger confidence,
stronger anchor, residual texture fusion, or another topK selector.  The next
mainline should strengthen **cross-view matchability supervision**:

- hard scene negatives from repeated corridor/room structures;
- multi-view NVS correspondence/warping beyond sparse LoFTR teacher points;
- a pose-energy sharpness objective that couples positive GT render, hard
  same-scene negatives, and ranking, instead of optimizing Fisher magnitude
  alone;
- only then revisit low-LR map-side adaptation.

## Non-Mainline / Future

- Full retrieval/deploy relocalization still needs external retrieval baselines
  and cross-scene validation before it can be a paper claim.
- Solver ablations (`MLP`, `trans-WLS`, `full-WLS`, `feature-metric GN`) remain
  important but are not the immediate q50 feature bottleneck.
- Feature selection visual masks are not a current architectural output because
  POFD was simplified to one localization feature.  Visual evidence should focus
  on feature maps, confidence if enabled, correlation entropy, and failure cases.
