# 2026-05-16 POFD Stage4 Single-Render Continuous CPR

## Goal

Implement the latest expert pivot from multi-candidate render-and-rank toward:

```text
render once -> match query/render POFD features -> solve 2D-3D pose update -> iterate
```

Multi-candidate rendering remains useful for Stage3a diagnosis and oracle/cache construction, but it is no longer the intended paper main inference path.

## Implemented

- `feature_extract/tools/train_nvs_pose_feature_adapter.py`
  - Added `pair_matcher_single_render_correspondences`.
    - Uses the existing asymmetric `PairConditionedLocalMatcher`.
    - Produces query-centered correspondences: `query_xy`, `render_xy`, `world_points`, `offset_logits`, `expected_offset`, `argmax_offset`, confidence, and valid mask.
    - Supports `score_feature_hw`, sparse stride, render valid masks, and chunked matcher inference.
  - Added `robust_pose_update_from_correspondences`.
    - Pure-torch robust 2D-3D pose solver.
    - Solves left-multiplied SE(3) updates with Huber reweighting, LM damping, min correspondence count, and per-iteration update clamps.
    - Returns final pose, total delta, inlier count, reprojection error, success flag, and conditioning proxy.
  - Added `--stage4-eval-only`.
    - Runs one-render-per-iteration refinement from `pose_init` when a cache provides it, otherwise from the configured controlled perturbation.
    - Reports initial/final cost, translation/rotation, success metrics, render count, reprojection residual, and solver diagnostics.
  - Fixed Stage4 cache initialization.
    - The first implementation ignored cached `pose_init` and always used `candidate_center_pose`, which made q10/q25/q50 probes identical. Stage4 eval now uses cached `pose_init` in `candidate_bank_mode=cache`.
  - Added `stage4_pair_match_flow_loss` and `--stage4-pair-match-flow-weight`.
    - Supervises query-centered nonzero offsets by projecting candidate-render world points into the GT query view.
    - Resizes the query feature to the render/geometry score grid, so low-resolution q50 score geometry is supported.
  - Added Python 3.8 compatibility shims/fixes for CLI import/runtime blockers.
  - Added Stage4d render-once virtual trust-region scoring.
    - Scores local virtual pose candidates from one render by projectively sampling query features at reprojected render world points.
    - Reports virtual accept fraction, score gap, identity score, best score, and valid fraction.
  - Added deployable Stage4 update diagnostics and accept gates.
    - `stage4_pose_update_diagnostics` reports proposed GT cost gain, correction cosine, camera-center step, and rotation step.
    - `stage4_accept_pose_updates` gates accepted solver updates by success, reprojection, optional max camera-center step, and optional max rotation step without using GT.
    - Fixed the diagnostic rotation unit: `pose_error_tensors` already returns degrees, so Stage4 no longer multiplies by `180/pi` a second time.
  - Added exact PMED GeoNCE mode for selected hard negatives.
    - `stage4_pair_match_flow_positive_only_ce` applies offset CE only to the positive/oracle candidate.
    - Hard-negative candidates are still scored for the cross-candidate margin, but are no longer taught their own offset CE target when this mode is enabled.
  - Added richer candidate dump diagnostics.
    - `--eval-failure-dump-top-n` controls how many score/cost-ranked candidates are written per row.
    - Dump rows can now include each candidate's translation/rotation step from the init pose.
    - Fixed candidate-delta rotation dump units: `pose_error_tensors` already returns degrees, so `delta_rot_deg` is no longer multiplied by `180/pi` a second time.
  - Added Stage4 proposal virtual acceptance diagnostics/gate.
    - `--stage4-proposal-virtual-gate-enabled`
    - `--stage4-proposal-virtual-min-score-gap`
    - `--stage4-proposal-virtual-min-valid-frac`
    - Scores the solver proposal against the identity/current pose using the same single render, then optionally requires `score(proposal)-score(identity)` to pass a deployable threshold.

- Configs:
  - `feature_extract/configs/pofd_stage4_single_render_q10_val128_eval.yaml`
  - `feature_extract/configs/pofd_stage4_single_render_q25_val128_eval.yaml`
  - `feature_extract/configs/pofd_stage4_single_render_q50_val128_eval.yaml`
  - `feature_extract/configs/pofd_stage4_single_render_realinit_netvlad_eval.yaml`
  - `feature_extract/configs/pofd_stage4_pairflow_mixed_10_25_50_shuf.yaml`
  - `feature_extract/configs/pofd_stage4_pairflow_q50_shuf.yaml`

- Tests:
  - `test_pair_matcher_single_render_correspondences_tracks_query_to_render_offset`
  - `test_robust_pose_update_from_correspondences_recovers_translation_pose`
  - `test_stage4_single_render_eval_uses_cached_pose_init_for_cache_mode`
  - `test_stage4_pair_match_flow_loss_trains_nonzero_query_to_render_offset`
  - `test_stage4_pair_match_flow_loss_resizes_query_to_render_resolution`
  - `test_stage4_virtual_pose_energy_scores_prefers_projective_shift_alignment`
  - `test_stage4_pose_update_diagnostics_reports_direction_and_cost_gain`
  - `test_stage4_accept_pose_updates_rejects_large_pose_step`
  - `test_stage4_pair_match_flow_loss_positive_only_ce_skips_hard_negative_ce`

## Commands

Targeted regression:

```bash
pytest tests/test_nvs_pose_feature_adapter.py::test_pair_matcher_single_render_correspondences_tracks_query_to_render_offset \
  tests/test_nvs_pose_feature_adapter.py::test_robust_pose_update_from_correspondences_recovers_translation_pose -q
```

Stage4 q50 eval:

```bash
CUDA_VISIBLE_DEVICES=0 python -m feature_extract.tools.train_nvs_pose_feature_adapter \
  --config feature_extract/configs/pofd_stage4_single_render_q50_val128_eval.yaml \
  --checkpoint /root/ICLPose/result/result/feature_extract/cpr_phase6a2_metric_align_candidate_poseaware_sharedproj_w005_quick_b4/checkpoints/best.pth \
  --map-checkpoint /root/ICLPose/result/result/feature_extract/merged_checkpoints/adapter_on_safemap_top4_model_safe_decoder_map.pth \
  --resume-adapter result/result/feature_extract/pofd_stage3a_val128_r12_continue_b2_noamp_s40_20260516/checkpoints/best.pth \
  --stage4-eval-only \
  --eval-max-samples 128 \
  --batch-size 1 \
  --no-amp \
  --out-dir result/result/feature_extract/eval_pofd_stage4_q50_val128_20260516
```

Parallel controlled regression:

```bash
CUDA_VISIBLE_DEVICES=0 python -m feature_extract.tools.train_nvs_pose_feature_adapter ...pofd_stage4_single_render_q50_val128_eval.yaml ...
CUDA_VISIBLE_DEVICES=1 python -m feature_extract.tools.train_nvs_pose_feature_adapter ...pofd_stage4_single_render_q25_val128_eval.yaml ...
```

Then run q10 after the first GPU frees.

## Promote / Continue / Stop

Promote Stage4 as paper main path only if:

```text
OldHospital q50 val128 final pred_cost <= 0.18m
q10/q25 do not regress versus Stage3a
render_count <= 5
full validation gap <= 15mm
real-init deploy evaluation improves over the initializer
at least one public benchmark is competitive with GS-CPR / GS-SMC style baselines
```

Continue if q50 is `0.18m~0.20m`, render count is low, real-init median error improves by at least 20%, and solver diagnostics are stable.

Stop and pivot if q50 remains above `0.21m`, three controlled configs improve less than 10mm over Stage3a, solver failure is frequent, q10/q25 regress, or real/public-init evaluation does not improve over the initializer.

## Current Risks

- Stage4 uses query-centered correspondences to match the trained asymmetric pair matcher. This avoids retraining, but a future PMED/GeoNCE update should supervise the matcher directly for continuous CPR correspondence quality.
- The first Stage4 eval may underperform Stage3a because Stage3a was trained for candidate score surfaces, not for direct 2D-3D pose solving.
- Public SOTA claims remain unproven until ACE/GLACE/Marepo real-init protocols, POFD refinement on top of public initializers, and GS-CPR/GS-SMC/GSFF comparisons are added. HLoc now provides Cambridge initializer/baseline pose tables, but not a POFD SOTA result.

## 2026-05-16 Smoke Results

Targeted tests pass:

```text
pytest tests/test_feature_extract_checkpoint_io.py tests/test_nvs_pose_feature_adapter.py -q
90 passed
```

One-sample q50 probes showed that direct continuous updates are not ready to promote:

| probe | init cost | final cost | accepted | conclusion |
|---|---:|---:|---:|---|
| pair matcher, ungated | 0.259m | 1.072m | 1/1 | destructive update |
| pair matcher, min-conf/top64, min12 | 0.259m | 0.309m | 1/1 | still harmful |
| local-corr render-to-query, min-conf/top64, min12 | 0.259m | 0.322m | 1/1 | still harmful |
| pair matcher, safe default gate | 0.259m | 0.259m | 0/1 | safe fallback |

Cache-init audit result:

```text
Root cause: Stage4 eval ignored cached pose_init and initialized from candidate_center_pose.
Fix: stage4_initial_pose_from_batch now uses batch["pose_init"] for candidate_bank_mode=cache.
```

After the fix, four-sample q10/q25/q50 controlled probes are basin-specific. All still reject updates, but the initialization costs now match the cache basin:

```text
Stage3a checkpoint, safe gate, 3 renders:
q10 init=final=0.1035m, accept=0.0, reproj=4.02px, valid=0.167
q25 init=final=0.2586m, accept=0.0, reproj=4.23px, valid=0.158
q50 init=final=0.5168m, accept=0.0, reproj=5.06px, valid=0.150
```

Stage4 pair-flow training probes:

```text
1-step real-path smoke:
  result/result/feature_extract/pofd_stage4_pairflow_smoke_s1_cachefix_20260516
  stage4_pair_match_flow_points=1934, target_offset=1.97px, loss=4.11

20-step mixed q10/q25/q50 probe:
  result/result/feature_extract/pofd_stage4_pairflow_mixed_b1_s20_20260516
  candidate-selection eval pred_cost=0.066m on 8 samples

20-step q50-focused probe:
  result/result/feature_extract/pofd_stage4_pairflow_q50_b1_s20_20260516
  candidate-selection eval pred_cost=0.129m on 8 samples
```

Single-render Stage4 eval after 20-step pair-flow training:

```text
mixed checkpoint on q10: init=final=0.1035m, accept=0.0, reproj=0.76px, valid=0.004
mixed checkpoint on q25: init=final=0.2586m, accept=0.0, reproj=0.44px, valid=0.002
q50 checkpoint on q50: init=final=0.5168m, accept=0.0, reproj=3.86px, valid=0.160
```

Interpretation:

```text
Stage4 infrastructure is runnable, cache-correct, and now has explicit nonzero
offset supervision. However, the current 20-step probes still do not produce
enough high-quality inliers for accepted continuous pose updates. q50 residuals
improved but remain above the 3px acceptance gate; mixed q10/q25 collapsed the
high-confidence valid fraction. Stage4 is not ready to promote.
```

## 2026-05-17 External Iterative Refiner Follow-Up

Implementation:

```text
pose_refine/tools/eval_render_loftr_refine.py now supports:
  --iterations N

Each iteration renders at the current deploy pose, runs LoFTR+PnP, and updates
the pose for the next render. If a later iteration fails, the exported pose
cache keeps the last successful pose instead of discarding it. The cache now
also records scalar iteration diagnostics:
  refine_iterations_requested
  refine_attempted_iterations
  refine_successful_iterations
  refine_last_success_iteration
```

Tests:

```text
tests/test_eval_render_loftr_refine.py
  test_record_from_iterative_refinement_results_keeps_last_success_after_failure
  test_pose_cache_payload_from_records_exports_iterative_refinement_scalars

Targeted/broad regression:
  pytest tests/test_eval_render_loftr_refine.py -q -> 10 passed
  broad targeted localization/init/refine suite -> 150 passed
  py_compile + git diff --check -> passed
```

Full182 OldHospital real-init results, LoFTR conf=0.3/reproj=4px:

| cache | median trans | mean trans | improved vs init | interpretation |
|---|---:|---:|---:|---|
| one-pass render-at-init | 156.3mm | 273.4mm | 133/182 | strongest median/external baseline |
| iter2 render-at-current | 175.5mm | 264.5mm | 123/182 | lower mean, worse median/robustness |
| iter3 render-at-current | 181.9mm | 264.3mm | 123/182 | lower mean, worse median/robustness |

No-GT quality-gating diagnostic:

```text
iter2 vs one-pass: 70 better / 112 worse, mean delta -8.9mm, median delta +6.9mm
iter3 vs one-pass: 66 better / 116 worse, mean delta -9.0mm, median delta +7.1mm
inlier/raw-match deltas have weak positive correlation with error delta
  corr ~= 0.16-0.18
simple inlier/raw thresholds do not recover one-pass median or 10cm success
```

No-GT pose-step gating diagnostic:

```text
feature_retrieval/tools/gate_refined_init_cache.py now includes
gate_refined_init_cache_by_pose_step() and a matching CLI mode enabled by
--candidate_refined_cache / --step_candidate_refined_cache.

The useful rule is deploy-style and does not use GT: keep the one-pass
render-at-init LoFTR pose by default, but switch to iter3 only when the
one-pass -> iter2 camera-center step is >= 0.20m. This selects iter3 on
35/182 samples and keeps one-pass on 147/182 samples.

Export:
result/result/feature_extract/pose_init_exports/oldhospital_renderatinit_loftr_refined_top50qf_full182_conf03_r4_iter3_stepgate_iter2d12ge020_20260517.npz

Config:
feature_extract/configs/pofd_stage4_single_render_realinit_renderloftr_stepgated_full182_eval.yaml
```

| cache | median trans | mean trans | median rot | mean rot | succ@10/25/50cm | interpretation |
|---|---:|---:|---:|---:|---:|---|
| one-pass render-at-init | 156.3mm | 273.4mm | 0.248deg | 0.387deg | 0.335 / 0.637 / 0.868 | strongest median external cache |
| iter3 render-at-current | 181.9mm | 264.3mm | 0.234deg | 0.366deg | 0.269 / 0.654 / 0.890 | lower mean, worse median/10cm |
| iter3 pose-step gated by one->iter2 >=0.20m | 159.6mm | 256.2mm | 0.225deg | 0.360deg | 0.346 / 0.654 / 0.896 | best external-cache mean, near one-pass median |

Stage4 all-margin q50 on the step-gated external cache:

```text
result/result/feature_extract/eval_pofd_stage4_stepgated_refined_full182_explicit_20260517/stage4_eval_summary.json

stage4_init_cost_m = 0.2568735
stage4_pred_cost_m = 0.2568735
stage4_cost_gain_m = 0.0
stage4_init_trans_m = stage4_pred_trans_m = 0.2562460
stage4_init_rot_deg = stage4_pred_rot_deg = 0.3595561
stage4_solver_success = 0.0
stage4_update_accept_frac = 0.0
stage4_render_count = 3.0
stage4_pred_success_5/10/25/50cm = 0.104 / 0.346 / 0.654 / 0.896
```

The Stage4 POFD updater remains a safe no-op on this stronger external init;
the improvement comes entirely from the external LoFTR pose-step cache gate.

Continuous featuremetric GN sanity on the one-pass refined cache:

```text
s64, max_iters=10, damping=0.01:
  init 85.5mm median -> refined 119.1mm median
  improved 1/64

s64, max_iters=5, damping=0.1:
  init 85.5mm median -> refined 114.9mm median
  improved 1/64

s64, max_iters=5, damping=1.0:
  init 85.5mm median -> refined 117.1mm median
  improved 1/64
```

Decision:

```text
The external iterative LoFTR path is implemented and reproducible, but it is not
a clean promote path: extra iterations reduce outlier mean slightly while
hurting the median and most samples. A no-GT pose-step gate recovers a useful
external-cache trade-off, improving the mean from 273.4mm to 256.2mm while
keeping the median close to one-pass (159.6mm vs 156.3mm), but the gain is
still from external LoFTR rather than POFD. Existing DCFF featuremetric GN also
regresses the strong external refined init, so the remaining bottleneck is
correspondence/uncertainty quality rather than only solver iteration count.
Use one-pass LoFTR as the median-oriented teacher and the pose-step-gated cache
as the mean-oriented external baseline; do not promote iter2/iter3, FM-GN, or
the no-op Stage4 pass as the paper main method.
```

Next required implementation step:

```text
Improve correspondence training before full validation:
1. add confidence calibration / coverage regularization so q10/q25 retain enough inliers;
2. tune pair-flow weight and min-confidence jointly with q50 residual reduction;
3. run q10/q25/q50 acceptance diagnostics at 20/80/200 steps;
4. only then run full val128 and real-init/public protocol comparisons.
```

## 2026-05-16 PMED / Margin Follow-Up

Additional implementation:

```text
Added Stage4 pair-flow hard-negative margin:
  --stage4-pair-match-flow-margin-weight
  --stage4-pair-match-flow-margin

Added PMED-style candidate selection and cross-candidate logit diagnostics:
  --stage4-pair-match-flow-candidate-mode
  --stage4-pair-match-flow-hard-negative-min-cost-gap-m
  --stage4-pair-match-flow-cross-candidate-margin-weight
  --stage4-pair-match-flow-cross-candidate-margin
  --stage4-pair-match-flow-positive-only-ce

Added pair-matcher score-map aggregation probe:
  --pair-matcher-score-pooling {mean,topk_mean}
  --pair-matcher-score-topk-fraction

Added q50 candidate failure dump instrumentation:
  --eval-failure-dump-path
  --eval-failure-dump-max-rows
  --eval-failure-dump-min-gap-m
  --eval-failure-dump-include-correct
  --eval-failure-dump-top-n

Added signed candidate-delta score prior:
  --candidate-delta-trans-score-penalty
  --candidate-delta-rot-score-penalty
  --candidate-delta-trans-score-reward
  --candidate-delta-rot-score-reward

Added optional Stage4d render-once virtual trust-region warm start:
  --stage4-virtual-trust-region-enabled
  --stage4-virtual-trust-region-trans-cm
  --stage4-virtual-trust-region-rot-deg
  --stage4-virtual-trust-region-max-candidates
  --stage4-virtual-trust-region-min-score-gap

Main pair-flow configs keep all-candidate margin as the safer default.
PMED hard-negative variants are isolated in:
  feature_extract/configs/pofd_stage4_pmed_mixed_10_25_50_shuf.yaml
  feature_extract/configs/pofd_stage4_pmed_q50_shuf.yaml
Correction-reward eval configs are isolated in:
  feature_extract/configs/pofd_stage4_pairflow_delta_reward_q10_val128_eval.yaml
  feature_extract/configs/pofd_stage4_pairflow_delta_reward_q25_val128_eval.yaml
  feature_extract/configs/pofd_stage4_pairflow_delta_reward_q50_val128_eval.yaml
```

Four-sample Stage4 CPR probes:

| checkpoint / gate | basin | init | final | gain | any accept | conclusion |
|---|---:|---:|---:|---:|---:|---|
| mixed margin s20, safe | q10 | 0.103m | 0.103m | 0.000m | 0.00 | safe fallback |
| mixed margin s20, min12/tiny trust | q10 | 0.103m | 0.163m | -0.059m | 0.50 | reject for q10 |
| mixed margin s20, min12/tiny trust | q25 | 0.259m | 0.240m | +0.018m | 0.50 | small positive |
| q50 margin s20, min12/tiny trust | q50 | 0.517m | 0.494m | +0.023m | 0.25 | small positive |
| q50 all-margin s80 best, min12/tiny trust | q50 | 0.517m | 0.488m | +0.029m | 0.50 | best Stage4 signal |
| mixed all-margin s80 best, min12/tiny trust | q25 | 0.259m | 0.292m | -0.033m | 1.00 | over-updates, reject |
| PMED cross-candidate s20 | q50 | 0.517m | 0.522m | -0.005m | 0.25 | not promoted |

Candidate-ranking val128 checks:

| checkpoint | basin | pred_cost | top1 | spearman | oracle gap | status |
|---|---:|---:|---:|---:|---:|---|
| Stage3a multiscale baseline | q50 | 0.233m | 0.703 | 0.526 | 0.104m | current mean-pooling reference |
| Stage3a + topk_mean 0.2 | q50 | 0.290m | 0.492 | 0.505 | 0.160m | worse; reject aggregation |
| q50 all-margin s80 best (step 40) | q50 | 0.231m | 0.695 | 0.509 | 0.101m | worse than Stage3a |
| q50 all-margin s80 best + topk_mean 0.2 | q50 | 0.306m | 0.461 | 0.516 | 0.177m | worse; reject aggregation |
| q50 all-margin s80 final (step 80) | q50 | 0.256m | 0.625 | 0.476 | 0.127m | worse |
| mixed margin s20 | q25 | 0.128m | 0.648 | 0.478 | 0.064m | q25 candidate ranking not promoted |

PMED selected hard-negative 20-step smoke checks:

| checkpoint / setting | basin mix | pred_cost | top1 | spearman | flow points | pos-hard logit gap | status |
|---|---:|---:|---:|---:|---:|---:|---|
| PMED cross-candidate s20 | q50 | 0.129m | 1.000 | 0.390 | 220 | -0.069 | best selected-hard-negative smoke |
| PMED positive-only CE, xcw 0.25 | q50 | 0.153m | 0.938 | 0.503 | 111 | -0.066 | worse ranking |
| PMED positive-only CE, xcw 1.0 | q50 | 0.153m | 0.938 | 0.503 | 111 | -0.066 | no recovery |
| D1 cost-listwise only | q50 | 0.173m | 0.875 | 0.508 | 0 | 0.000 | isolated listwise branch worse |
| D2 GeoNCE only, positive-only CE | q50 | 0.153m | 0.938 | 0.499 | 111 | -0.067 | isolated GeoNCE matches positive-only no-go |
| D3 listwise + GeoNCE, positive-only CE | q50 | 0.164m | 0.906 | 0.500 | 111 | -0.066 | interaction worsens q50 |
| PMED cross-candidate s20 | q10/q25/q50 | 0.066m | 1.000 | 0.594 | 236 | -0.048 | best selected-hard-negative smoke |
| PMED positive-only CE, xcw 0.25 | q10/q25/q50 | 0.095m | 0.906 | 0.568 | 120 | -0.040 | worse ranking |
| PMED positive-only CE, xcw 1.0 | q10/q25/q50 | 0.095m | 0.906 | 0.569 | 120 | -0.039 | no recovery |
| D1 cost-listwise only | q10/q25/q50 | 0.113m | 0.844 | 0.546 | 0 | 0.000 | isolated listwise mixed worse |
| D2 GeoNCE only, positive-only CE | q10/q25/q50 | 0.072m | 1.000 | 0.577 | 120 | -0.050 | mixed recovers top1 but trails old cross-candidate |
| D3 listwise + GeoNCE, positive-only CE | q10/q25/q50 | 0.095m | 0.906 | 0.567 | 120 | -0.047 | listwise interaction again hurts |

Recommendation-matrix full-val follow-up:

| checkpoint / setting | eval | pred_cost | top1 | spearman | oracle gap | succ@25cm/10deg | status |
|---|---:|---:|---:|---:|---:|---:|---|
| Stage3a r12 | q50 val128 | 0.233m | 0.703 | 0.526 | 0.104m | 0.719 | S0 r12 reference |
| Stage3a r16 | q50 val128 | 0.223m | 0.719 | 0.550 | 0.093m | 0.742 | S0 best score-path reference |
| old cross-candidate PMED r12 | q50 val128 | 0.220m | 0.727 | 0.519 | 0.091m | 0.758 | small gain, not promotable |
| old cross-candidate PMED r16 | q50 val128 | 0.225m | 0.719 | 0.534 | 0.095m | 0.742 | r16 no gain |
| D3 listwise + GeoNCE r12 | q50 val128 | 0.226m | 0.711 | 0.518 | 0.096m | 0.742 | below Stage3a r16 |
| D4 D3 + r16 eval-only | q50 val128 | 0.227m | 0.719 | 0.533 | 0.097m | 0.734 | r16 no gain |

D5 controlled-basin regression for D3:

| checkpoint / setting | eval | pred_cost | top1 | spearman | oracle gap | succ@10cm/5deg | status |
|---|---:|---:|---:|---:|---:|---:|---|
| D3 listwise + GeoNCE r12 | q10 val128 | 0.046m | 0.742 | 0.597 | 0.020m | 0.977 | q10 cost/success target met |
| D3 listwise + GeoNCE r12 | q25 val128 | 0.111m | 0.758 | 0.496 | 0.047m | 0.758 | q25 target missed |

Delta-prior rerank probe on old cross-candidate PMED q50 val128:

| candidate delta trans penalty | pred_cost | top1 | spearman | oracle gap | succ@25cm/10deg | status |
|---:|---:|---:|---:|---:|---:|---|
| 0.0 | 0.220m | 0.727 | 0.519 | 0.091m | 0.758 | reference |
| 0.1 | 0.235m | 0.688 | 0.517 | 0.105m | 0.719 | worse |
| 0.2 | 0.243m | 0.672 | 0.518 | 0.114m | 0.688 | worse |
| 0.5 | 0.286m | 0.555 | 0.518 | 0.157m | 0.570 | much worse |
| 1.0 | 0.343m | 0.398 | 0.513 | 0.214m | 0.414 | destructive |

Correction-reward rerank probe on old cross-candidate PMED:

| trans reward | eval | pred_cost | top1 | spearman | oracle gap | success | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | q50 val128 | 0.220m | 0.727 | 0.519 | 0.091m | succ@25=0.758 | reference |
| 0.50 | q50 val128 | 0.196m | 0.805 | 0.522 | 0.066m | succ@25=0.828 | crosses selection gates except spearman |
| 1.00 | q50 val128 | 0.176m | 0.844 | 0.517 | 0.047m | succ@25=0.867 | stronger selection, lower spearman |
| 1.50 | q50 val128 | 0.171m | 0.859 | 0.509 | 0.042m | succ@25=0.883 | stronger selection, lower spearman |
| 2.00 | q50 val128 | 0.160m | 0.898 | 0.500 | 0.030m | succ@25=0.914 | best q50 selection, rank calibration still weak |
| 2.00 | q25 val128 | 0.073m | 0.961 | 0.487 | 0.008m | succ@10=0.961 | q25 target recovered |
| 2.00 | q10 val128 | 0.028m | 0.984 | 0.593 | 0.002m | succ@5=0.984 | q10 remains strong |

Full182 controlled-cache reward validation used newly completed q50/q25/q10
tail shards with the same no-exact lattice definitions as the original val128
runs. An earlier shard attempt used mismatched direction fractions and was
discarded after the cache stats failed to match the val128 setup. The accepted
full182 shuffled caches are:

```text
result/result/feature_extract/pose_init_exports/oldhospital_local_lattice_renderloftr_q50cm10deg_top16_noexact_val182_shuf20260517.npz
result/result/feature_extract/pose_init_exports/oldhospital_local_lattice_renderloftr_q25cm5deg_top16_noexact_val182_shuf20260517.npz
result/result/feature_extract/pose_init_exports/oldhospital_local_lattice_renderloftr_q10cm2deg_top16_noexact_val182_shuf20260517.npz
```

Full182 eval configs:

```text
feature_extract/configs/pofd_stage4_pairflow_delta_reward_q50_full182_eval.yaml
feature_extract/configs/pofd_stage4_pairflow_delta_reward_q25_full182_eval.yaml
feature_extract/configs/pofd_stage4_pairflow_delta_reward_q10_full182_eval.yaml
```

Full182 correction-reward summary from `eval_only_summary.json`:

| trans reward | eval | pred_cost | top1 | spearman | oracle gap | success | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2.00 | q50 full182 | 0.179m | 0.874 | 0.481 | 0.050m | succ@25=0.890 | cost/top1/success hold, rank calibration fails |
| 2.00 | q25 full182 | 0.071m | 0.967 | 0.487 | 0.006m | succ@10=0.967 | q25 protected, rank calibration weak |
| 2.00 | q10 full182 | 0.028m | 0.984 | 0.601 | 0.002m | succ@5=0.984 | q10 remains strong |

The full182 check preserves the useful selection behavior, especially q10/q25,
but it does not change the paper decision: q50 rank calibration remains below
the `spearman >= 0.58` gate, and the method is still a hand-coded cached-step
reward over a multi-candidate cache rather than a learned single-render
continuous refinement path.

Failure-dump checks on q50 val128:

| checkpoint | failure rows | mean failure gap | severe >=20cm | selected score > oracle score | mean selected-oracle score gap | diagnosis |
|---|---:|---:|---:|---:|---:|---|
| Stage3a multiscale baseline | 38/128 | 0.351m | 36 | 38 | 0.359 | wrong candidates remain score-dominant |
| q50 all-margin s80 best | 39/128 | 0.333m | 35 | 39 | 0.592 | slightly lower failure cost, but stronger wrong-candidate confidence |
| old cross-candidate PMED r12 | 35/128 | 0.331m | 31 | 35 | 0.569 | oracle usually inside top-score-5 but under-ranked |
| D3 listwise+GeoNCE r12 | 37/128 | 0.334m | 33 | 37 | 0.548 | same 35 shared wrong predictions as old PMED |

All-sample/failure-dump root-cause notes:

| check | old cross-candidate PMED | D3 listwise+GeoNCE | interpretation |
|---|---:|---:|---|
| shared q50 failure rows | 35 | 35 | D3 does not change selected candidate on shared failures |
| oracle in top-score-5 on failures | 31/35 | 33/37 | scorer often finds the right neighborhood |
| best cost inside top-score-5 | 0.142m | 0.141m | near-oracle candidate is usually close in score order |
| selected delta from init, all val128 | 0.291m | 0.285m | selected candidates under-correct from the 0.517m init |
| oracle delta from init, all val128 | 0.375m | 0.375m | oracle requires a larger correction step |

Conclusion: positive delta penalty amplified the under-correction bias. The q50
cache needs a correction-magnitude reward, not a "prefer smaller update from
init" prior.

Top-16 all-sample dump calibration sweep on old cross-candidate PMED q50:

| offline transform family | best pred_cost | best top1 | best spearman | best oracle gap | interpretation |
|---|---:|---:|---:|---:|---|
| baseline raw score | 0.220m | 0.727 | 0.519 | 0.091m | reference |
| linear init-step reward | 0.160m | 0.898 | 0.500 | 0.030m | fixes selection, hurts rank calibration |
| raw quadratic step prior, best spearman | 0.248m | 0.617 | 0.532 | 0.119m | higher spearman but unusable selection |
| z/rank-normalized step/peak priors, selection-gated | 0.161m | 0.875 | 0.527 | 0.031m | best simple calibration trade-off |
| any scanned transform with spearman >= 0.55 | none | none | none | none | simple non-learned step prior is insufficient |

The sweep covered 377,620 transforms built from raw score, row-zscored score,
score rank, translation step, step rank, quadratic step, and peaked step priors.
No transform reached the q50 `spearman >= 0.58` promote target. This rules out
promoting a hand-coded correction prior as the paper solution; it is only a
controlled-cache reranker baseline.

The top-16 dumps were regenerated after fixing candidate-delta rotation units:

```text
result/result/feature_extract/pofd_stage4_pmed_xcand_q50_val128_alldump16_r12_fixedrot_20260516/candidate_all16_dump.jsonl
result/result/feature_extract/pofd_stage4_pairflow_delta_reward_q50_val128_alldump16_r2_fixedrot_20260516/candidate_all16_dump.jsonl
```

The selection metrics are unchanged from the original runs. The diagnostic
`delta_rot_deg` range is now `0.028-7.500deg` instead of hundreds of degrees.

Offline calibrated score-head probe on q50 all16 dumps:

| source scores / features | split protocol | pred_cost | top1 | spearman proxy | oracle gap | succ@25cm/10deg | interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| old PMED raw score | even/odd readout | 0.220m | 0.727 | 0.554 | 0.091m | 0.758 | reference in dump metric |
| old PMED raw + `2.0 * delta_trans` | fixed transform | 0.160m | 0.898 | 0.534 | 0.030m | 0.914 | good selection, lower rank calibration |
| old PMED score+delta ridge | even/odd CV | 0.277m | 0.562 | 0.575 | 0.148m | 0.602 | improves rank proxy but unusable selection |
| reward score+delta ridge | even/odd CV | 0.205m | 0.758 | 0.580 | 0.075m | 0.789 | near rank target, misses cost/top1/success |
| cache PnP metrics ridge | even/odd CV | 0.129m | 1.000 | 0.999 | 0.000m | 1.000 | recovers oracle, but uses render-LoFTR/PnP cache evidence |

The cache-feature result is intentionally not a promoted POFD result. The
features are deployable within the existing multi-render render-LoFTR/PnP
candidate generator, but using them as the main selector would collapse the
contribution back to a LoFTR/PnP selection baseline. The useful diagnostic is
that q50 cache candidates contain enough external PnP evidence to identify the
oracle; the POFD pair-matcher score surface still does not reproduce that
evidence without importing the multi-render cache metrics.

Offline POFD-only score-neighborhood aggregation probe:

| aggregation objective | split protocol | pred_cost | top1 | spearman proxy | oracle gap | succ@25cm/10deg | interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| raw old PMED score | readout | 0.220m | 0.727 | 0.554 | 0.091m | 0.758 | reference in dump metric |
| raw + `2.0 * delta_trans` | fixed transform | 0.160m | 0.898 | 0.534 | 0.030m | 0.914 | good selection, lower rank calibration |
| geometry-neighborhood, selection-tuned | even/odd CV | 0.214m | 0.766 | 0.560 | 0.085m | 0.797 | small calibration gain, misses selection gates |
| geometry-neighborhood, rank-tuned | even/odd CV | 0.267m | 0.648 | 0.679 | 0.138m | 0.648 | rank improves by discarding useful selection signal |
| geometry-neighborhood, optimistic all-val selection tune | train=val diagnostic | 0.197m | 0.812 | 0.563 | 0.067m | 0.844 | still below q50 promote line |

This uses only POFD score, candidate-pose geometry, score ranks, and candidate
delta magnitudes. It does not use GT cost or render-LoFTR/PnP cache quality
fields at inference time. The result confirms that local score support can
smooth some isolated peaks, but the POFD score surface does not contain enough
deployable neighborhood evidence to satisfy both selection and rank-calibration
gates. The rank-tuned variant reaches a high Spearman proxy by suppressing
selection quality, so it is not a paper-grade score head.

Stage4d render-once virtual trust-region probes:

| checkpoint / setting | basin | init | final | gain | virtual accept | virtual score gap | status |
|---|---:|---:|---:|---:|---:|---:|---|
| all-margin q50 s80 best, min gap 0.02 | q50 | 0.517m | 0.517m | 0.000m | 0.00 | 0.011 | safe no-op |
| all-margin q50 s80 best, min gap 0.00 | q50 | 0.517m | 0.570m | -0.054m | 1.00 | 0.011 | harmful |
| all-margin q50 s80 best, min gap 0.01 | q50 | 0.517m | 0.516m | +0.001m | 0.50 | 0.011 | negligible |
| mixed all-margin s80 best, min gap 0.00 | q25 | 0.259m | 0.257m | +0.002m | 1.00 | 0.003 | negligible |
| mixed all-margin s80 best, min gap 0.00 | q10 | 0.103m | 0.109m | -0.006m | 1.00 | 0.003 | regress |

Real-init NetVLAD/render-LoFTR smoke:

| setting | samples | init | final | gain | solver/update accept | virtual accept | status |
|---|---:|---:|---:|---:|---:|---:|---|
| default gate, no virtual | 10 | 0.072m | 0.072m | 0.000m | 0.00 | 0.00 | safe no-op |
| virtual min gap 0.01 | 10 | 0.072m | 0.134m | -0.062m | 0.00 | 1.00 | harmful; reject |
| min_points 24, 3 solver iters | 10 | 0.072m | 0.808m | -0.736m | 1.00 | 0.00 | destructive |
| min_points 12, 3 solver iters | 10 | 0.072m | 0.808m | -0.736m | 1.00 | 0.00 | same failure |
| min_points 24, 1 solver iter | 10 | 0.072m | 0.370m | -0.299m | 1.00 | 0.00 | first update already harmful |
| min_points 24, reproj <= 0.5px | 10 | 0.072m | 0.158m | -0.086m | 0.17 | 0.00 | stricter gate still harmful |
| 5cm/2deg step gate, default solver | 10 | 0.072m | 0.072m | 0.000m | 0.00 | 0.00 | safe no-op; rejects 37cm proposals |
| 3cm/1deg solver clamp + 5cm/2deg step gate | 10 | 0.072m | 0.072m | 0.000m | 0.00 | 0.00 | safe no-op; actual step still 13.6cm |
| 0.5cm/0.3deg solver clamp + 5cm/2deg step gate | 10 | 0.072m | 0.0716m | +0.0001m | 1.00 | 0.00 | negligible |
| 1cm/0.5deg solver clamp + 5cm/2deg step gate | 10 | 0.072m | 0.0758m | -0.004m | 0.80 | 0.00 | harmful |

Real-init step-gate val128 follow-up:

| setting | samples | init | final | gain | accept | success@10cm/5deg init->pred | proposal step | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5cm/0.3deg solver clamp + 5cm/2deg step gate | 128 | 0.3243m | 0.3221m | +0.0022m | 0.969 | 0.336 -> 0.344 | 2.42cm / 0.80deg | safe but too small |
| 0.25cm/0.2deg solver clamp + 3cm/1deg step gate | 128 | 0.3243m | 0.3230m | +0.0012m | 0.898 | 0.336 -> 0.344 | 1.21cm / 0.67deg | safe but too small |

Real-init top50 render-LoFTR PnP cache follow-up:

| setting | samples | init | final | gain | accept | success@10cm/5deg init->pred | success@25cm/10deg init->pred | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| top50 cache, 0.5cm/0.3deg clamp + 5cm/2deg gate | 128 | 0.3700m | 0.3677m | +0.0023m | 0.961 | 0.336 -> 0.336 | 0.617 -> 0.625 | safe but too small; rotation mean worsens |
| top50 cache, 0.25cm/0.2deg clamp + 3cm/1deg gate | 128 | 0.3700m | 0.3686m | +0.0015m | 0.914 | 0.336 -> 0.336 | 0.617 -> 0.617 | safe but too small; rotation mean worsens |
| top50 cache, virtual warm-start gap >=0.01, smoke32 | 32 | 0.0820m | 0.1306m | -0.0486m | virtual 1.000 | 0.781 -> 0.188 | 1.000 -> 1.000 | destructive; reject current virtual warm-start |
| top50 cache, virtual warm-start gap >=0.02, smoke32 | 32 | 0.0820m | 0.1205m | -0.0385m | virtual 0.812 | 0.781 -> 0.312 | 1.000 -> 1.000 | destructive; reject current virtual warm-start |
| top50 cache, proposal virtual gate >=0.000, smoke32 | 32 | 0.0820m | 0.0838m | -0.0017m | 0.688 | 0.781 -> 0.719 | 1.000 -> 1.000 | much safer, still not positive |
| top50 cache, proposal virtual gate >=0.005, smoke32 | 32 | 0.0820m | 0.0841m | -0.0021m | 0.656 | 0.781 -> 0.719 | 1.000 -> 1.000 | much safer, still not positive |
| top50 cache, proposal virtual gate >=0.010, smoke32 | 32 | 0.0820m | 0.0847m | -0.0027m | 0.625 | 0.781 -> 0.719 | 1.000 -> 1.000 | stricter but still negative |
| top50 cache, proposal virtual gate >=0.020, smoke32 | 32 | 0.0820m | 0.0838m | -0.0018m | 0.531 | 0.781 -> 0.750 | 1.000 -> 1.000 | safest threshold, still below init |
| top50 cache, iter3 no proposal gate, smoke32 | 32 | 0.0820m | 0.0836m | -0.0016m | any 1.000 / per-step 0.885 | 0.781 -> 0.656 | 1.000 -> 1.000 | repeated updates amplify wrong directions |
| top50 cache, iter3 proposal virtual gate >=0.020, smoke32 | 32 | 0.0820m | 0.0839m | -0.0019m | any 0.531 / per-step 0.198 | 0.781 -> 0.750 | 1.000 -> 1.000 | gate suppresses updates but still below init |
| top50 cache, iter1 + per-update dump, 0.5cm clamp | 128 | 0.3700m | 0.3677m | +0.0023m | 0.961 | 0.336 -> 0.336 | 0.617 -> 0.625 | reproduces small gain; dump enables confidence audit |
| top50 cache, iter1 + per-update dump, 0.25cm clamp | 128 | 0.3700m | 0.3686m | +0.0015m | 0.914 | 0.336 -> 0.336 | 0.617 -> 0.617 | reproduces smaller safe gain |
| top50 cache, expected offset, 0.5cm clamp | 128 | 0.3700m | 0.3697m | +0.0003m | 0.961 | 0.336 -> 0.320 | 0.617 -> 0.633 | worse than argmax; smoothing hurts direction |
| top50 cache, expected offset, 0.25cm clamp | 128 | 0.3700m | 0.3697m | +0.0003m | 0.820 | 0.336 -> 0.328 | 0.617 -> 0.633 | worse than argmax; not a proposal fix |
| top50 cache, local_corr argmax, 0.5cm clamp | 128 | 0.3700m | 0.3700m | +0.0000m | 0.000 | 0.336 -> 0.336 | 0.617 -> 0.617 | raw local corr no-op; valid frac 0.119, solver success 0.008 |
| top50 cache, local_corr argmax, 0.25cm clamp | 128 | 0.3700m | 0.3700m | +0.0000m | 0.000 | 0.336 -> 0.336 | 0.617 -> 0.617 | same no-op; not a replacement for pair matcher |

Output dirs:

```text
result/result/feature_extract/eval_pofd_stage4_realinit_renderloftr_top50_allmargin_q50_s80best_min24_iter1_smallstep0005_stepgate005_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_renderloftr_top50_allmargin_q50_s80best_min24_iter1_smallstep00025_stepgate003_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_renderloftr_top50_virtual001_smallstep0005_smoke32_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_renderloftr_top50_virtual002_smallstep0005_smoke32_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_renderloftr_top50_proposalvgap000_smallstep0005_smoke32_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_renderloftr_top50_proposalvgap005_smallstep0005_smoke32_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_renderloftr_top50_proposalvgap010_smallstep0005_smoke32_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_renderloftr_top50_proposalvgap020_smallstep0005_smoke32_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_renderloftr_top50_iter3_smallstep0005_smoke32_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_renderloftr_top50_iter3_proposalvgap020_smallstep0005_smoke32_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_top50_iter1_smallstep0005_dump_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_top50_iter1_smallstep00025_dump_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_top50_iter1_expected_smallstep0005_dump_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_top50_iter1_expected_smallstep00025_dump_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_top50_iter1_localcorr_argmax_smallstep0005_dump_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_top50_iter1_localcorr_argmax_smallstep00025_dump_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_smallstep0005_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_smallstep0005_iter2_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_smallstep0005_iter3_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_smallstep0005_iter2_gate1_val128_20260516
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_transonly0005_iter1_val128_20260517
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_transonly0005_iter2_val128_20260517
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_transonly0005_iter3_val128_20260517
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_transonly00025_iter2_val128_20260517
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_transonly0005_iter1_full182_20260517
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_transonly0005_iter2_full182_20260517
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_transonly0005_iter3_full182_20260517
result/result/feature_extract/eval_pofd_stage4_realinit_pairflow_top1_pm_only_w2_s80_transonly00025_iter2_full182_20260517
```

Real-init pair-flow top1 fine-tune val128 follow-up:

| setting | init | final | gain | pred rot | success@5cm/2deg | success@10cm/5deg | success@25cm/10deg | success@50cm/10deg | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PM-only w2, SE(3), iter1, 0.5cm clamp | 0.3700m | 0.3662m | +3.79mm | 0.72deg | 0.094 -> 0.109 | 0.336 -> 0.336 | 0.617 -> 0.633 | 0.781 -> 0.781 | small mean gain, rotation worsens |
| PM-only w2, SE(3), iter2, 0.5cm clamp | 0.3700m | 0.3647m | +5.36mm | 0.97deg | 0.094 -> 0.062 | 0.336 -> 0.352 | 0.617 -> 0.641 | 0.781 -> 0.773 | best SE(3) mean cost, not deployable |
| PM-only w2, SE(3), iter3, 0.5cm clamp | 0.3700m | 0.3642m | +5.87mm | 1.17deg | 0.094 -> 0.047 | 0.336 -> 0.336 | 0.617 -> 0.648 | 0.781 -> 0.773 | more updates amplify rotation error |
| PM-only w2, SE(3), iter2, strict gate | 0.3700m | 0.3658m | +4.28mm | 0.88deg | 0.094 -> 0.070 | 0.336 -> 0.352 | 0.617 -> 0.641 | 0.781 -> 0.781 | gate reduces gain, rotation still worse |
| PM-only w2, trans-only, iter1, 0.5cm clamp | 0.3700m | 0.3658m | +4.28mm | 0.53deg | 0.094 -> 0.109 | 0.336 -> 0.336 | 0.617 -> 0.633 | 0.781 -> 0.781 | rotation fixed, still small gain |
| PM-only w2, trans-only, iter2, 0.5cm clamp | 0.3700m | 0.3640m | +6.00mm | 0.53deg | 0.094 -> 0.086 | 0.336 -> 0.320 | 0.617 -> 0.641 | 0.781 -> 0.773 | best mean cost so far, success regresses |
| PM-only w2, trans-only, iter3, 0.5cm clamp | 0.3700m | 0.3660m | +4.03mm | 0.53deg | 0.094 -> 0.078 | 0.336 -> 0.320 | 0.617 -> 0.633 | 0.781 -> 0.773 | third update over-corrects |
| PM-only w2, trans-only, iter2, 0.25cm clamp | 0.3700m | 0.3657m | +4.29mm | 0.53deg | 0.094 -> 0.109 | 0.336 -> 0.336 | 0.617 -> 0.633 | 0.781 -> 0.781 | safer but no 10cm/50cm gain |

Real-init pair-flow top1 fine-tune full182 follow-up:

| setting | init | final | gain | pred rot | success@5cm/2deg | success@10cm/5deg | success@25cm/10deg | success@50cm/10deg | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PM-only w2, trans-only, iter1, 0.5cm clamp | 0.4403m | 0.4364m | +3.96mm | 0.68deg | 0.066 -> 0.077 | 0.247 -> 0.247 | 0.544 -> 0.560 | 0.742 -> 0.742 | safest full-set check, but tiny gain |
| PM-only w2, trans-only, iter2, 0.5cm clamp | 0.4403m | 0.4347m | +5.61mm | 0.68deg | 0.066 -> 0.060 | 0.247 -> 0.236 | 0.544 -> 0.560 | 0.742 -> 0.736 | full-set confirms small mean gain with success regressions |
| PM-only w2, trans-only, iter3, 0.5cm clamp | 0.4403m | 0.4362m | +4.19mm | 0.68deg | 0.066 -> 0.055 | 0.247 -> 0.236 | 0.544 -> 0.560 | 0.742 -> 0.736 | extra step loses mean gain and keeps regressions |
| PM-only w2, trans-only, iter2, 0.25cm clamp | 0.4403m | 0.4366m | +3.72mm | 0.68deg | 0.066 -> 0.077 | 0.247 -> 0.247 | 0.544 -> 0.560 | 0.742 -> 0.742 | safer but still only 0.8% relative gain |

Full182 median audit from the per-update dumps:

| setting | init median trans | final median trans | final median rot | interpretation |
|---|---:|---:|---:|---|
| PM-only w2, trans-only, iter1, 0.5cm clamp | 22.33cm | 21.65cm | 0.41deg | best median translation among 0.5cm variants, still small |
| PM-only w2, trans-only, iter2, 0.5cm clamp | 22.33cm | 22.10cm | 0.41deg | best mean cost, not best median |
| PM-only w2, trans-only, iter3, 0.5cm clamp | 22.33cm | 22.77cm | 0.41deg | median regresses after repeated steps |
| PM-only w2, trans-only, iter2, 0.25cm clamp | 22.33cm | 21.71cm | 0.41deg | safer median but smaller mean gain |

Full182 update-direction / acceptance audit:

| branch | base final gain | beneficial update frac | generated-update oracle gain | harmful generated loss | best held-out one-feature gate | interpretation |
|---|---:|---:|---:|---:|---:|---|
| trans-only, iter1, 0.5cm clamp | +3.96mm | 0.626 | +9.19mm | 5.23mm | not better than base | one update is safer but has too little oracle upside |
| trans-only, iter2, 0.5cm clamp | +5.61mm | 0.599 | +16.49mm | 10.88mm | ~+5.90mm/sample | oracle upper still only 3.7% relative; deployable gates are marginal |
| trans-only, iter3, 0.5cm clamp | +4.19mm | 0.551 | +22.46mm | 18.27mm | not better than base | generated oracle rises, but harmful updates rise faster |
| trans-only, iter2, 0.25cm clamp | +3.72mm | 0.624 | +9.10mm | 5.37mm | ~+3.88mm/sample | safer steps reduce both harm and upside |

The generated-update oracle is an upper diagnostic over the proposals that were
actually generated along the base trajectory; it is not a deployable alternate
trajectory. Even this optimistic accept/reject policy remains far below the
20% real-init improvement target. Simple held-out gates over reprojection,
match confidence/validity/offset, condition, solver inliers, proposal virtual
gap, and step size only marginally improve the row-level mean gain, confirming
that acceptance calibration is not the main missing component.

Per-update confidence audit from the new JSONL dumps:

| branch | no-update init | base gate | GT oracle accept current proposals | split-gated result |
|---|---:|---:|---:|---|
| 0.5cm clamp, val128 | 0.3700m | 0.3677m / +2.3mm | 0.3623m / +7.7mm | train-selected gates do not beat base on held-out even/odd or first/last splits |
| 0.25cm clamp, val128 | 0.3700m | 0.3686m / +1.5mm | 0.3661m / +3.9mm | train-selected gates do not beat base on held-out splits |
| expected offset 0.5cm, val128 | 0.3700m | 0.3697m / +0.3mm | 0.3636m / +6.4mm | worse reprojection and correction cosine than argmax |
| expected offset 0.25cm, val128 | 0.3700m | 0.3697m / +0.3mm | 0.3670m / +3.1mm | worse than argmax; reject offset smoothing |

The dump records deployable fields (`solver_success`, reprojection error,
condition, match valid/confidence/offset, proposal virtual gap/valid fraction,
step size) plus GT diagnostics (`before_cost`, `update_cost`,
`update_correction_cos`) for offline audits. The split sweep shows no stable
deployable confidence threshold. More importantly, even an oracle accept/reject
policy over the current solver proposals gives less than 8mm mean gain, so the
bottleneck is proposal direction quality, not only acceptance calibration.
Switching from argmax to expected offsets does not fix proposal quality: it
reduces the mean correction cosine from about 0.198 to 0.132, raises reprojection
error, and lowers base gain to 0.3mm.
Switching the Stage4 correspondence source to raw local correlation is worse:
the valid fraction falls to 0.119, solver success is only 0.008, no update passes
the existing gates, and the result is a pure no-op. Pair-matcher conditioning is
therefore necessary for usable correspondence density, but its learned offsets
are not yet directionally reliable enough for paper-grade continuous CPR.

Public/SOTA and real-init protocol audit:

| evidence | OldHospital/Hospital metric | interpretation |
|---|---:|---|
| GS-CPR, ACE init, Cambridge Hospital | 26cm / 0.38deg | public single-render 3DGS refinement reference |
| GS-SMC, DFNet init, Cambridge Hospital | 25cm / 0.39deg | public multi-candidate 3DGS refinement reference |
| GSVisLoc, Cambridge Hospital | 22cm / 0.42deg | public GS representation localization reference |
| HLoc, Cambridge Hospital | 15cm / 0.3deg | structure-based upper reference with reference images |
| local NetVLAD+render-LoFTR top50 PnP, full182 | 16.1-16.7cm / 0.24-0.27deg | strong deployable init/baseline, but multi-render and not POFD Stage4 |
| local oracle top50-selected CorrWLS, full182 | 7.8cm / 0.17deg | oracle selector upper bound; not deployable |
| local POFD Stage4 real-init val128 | 36.4cm mean cost after best trans-only refinement | current single-render POFD path is below public target |
| local POFD Stage4 real-init full182 | 21.7cm median trans / 0.41deg median rot; 43.5cm mean cost | best checked single-render median is still below local multi-render baselines and far from 20% initializer improvement |

Sources checked on 2026-05-16:

```text
GS-CPR: https://arxiv.org/abs/2408.11085
GS-SMC: https://arxiv.org/abs/2508.17876
UGS-Loc / uncertainty-aware 3DGS refinement: https://arxiv.org/abs/2603.16538
GSVisLoc: https://arxiv.org/abs/2508.18242
CVF GSVisLoc paper: https://openaccess.thecvf.com/content/ICCV2025W/CALIPOSE/papers/Khatib_Generalizable_Visual_Localization_for_Gaussian_Splatting_Scene_Representations_ICCVW_2025_paper.pdf
```

HLoc setup update on 2026-05-17:

```text
Official source: https://github.com/cvg/Hierarchical-Localization
Local clone: third_party/Hierarchical-Localization @ c13273b
Project env smoke: hloc 1.5 imports with pycolmap 3.12.5, h5py 3.11.0,
  cv2 4.13.0, torch 1.13.1+cu116, CUDA available on 2 GPUs; HLoc warns that
  pycolmap is below its declared >=3.13.0 requirement.
Python 3.10 smoke: conda env hloc-py310 imports hloc.localize_sfm with
  pycolmap 4.0.4, h5py 3.16.0, cv2 4.13.0 and no pycolmap warning.
Compatibility patches: hloc/match_features.py now supports
  HLOC_MATCH_NUM_WORKERS=0 to avoid DataLoader shared-memory bus errors, and
  hloc/triangulation.py now has pycolmap 3.12 path/database compatibility
  helpers for the local Python 3.8 run.
Official Cambridge model layout:
  /hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px,
  symlinked into result/result/hloc/cambridge_input.
ShopFacade HLoc compatibility result:
  result/result/hloc/cambridge_py38_compat/ShopFacade/results.txt,
  103/103 localized, median 0.042m / 0.206deg,
  5cm/5deg 59.22%, 25cm/2deg 96.12%, 50cm/5deg 99.03%.
OldHospital HLoc compatibility result:
  result/result/hloc/cambridge_py38_compat/OldHospital/results.txt,
  182/182 localized, median 0.144m / 0.309deg,
  5cm/5deg 8.24%, 25cm/2deg 66.48%, 50cm/5deg 86.81%.
KingsCollege HLoc compatibility result:
  result/result/hloc/cambridge_py38_compat/KingsCollege/results.txt,
  343/343 localized, median 0.114m / 0.210deg,
  5cm/5deg 15.74%, 25cm/2deg 73.47%, 50cm/5deg 91.25%.
GreatCourt HLoc compatibility result:
  result/result/hloc/cambridge_py38_compat/GreatCourt/results.txt,
  760/760 localized, median 0.175m / 0.107deg,
  5cm/5deg 12.11%, 25cm/2deg 64.21%, 50cm/5deg 80.39%.
StMarysChurch HLoc compatibility result:
  result/result/hloc/cambridge_py38_compat/StMarysChurch/results.txt,
  530/530 localized, median 0.075m / 0.224deg,
  5cm/5deg 32.08%, 25cm/2deg 95.66%, 50cm/5deg 99.25%.
```

HLoc pose-cache and rendered-RGB refiner bridge on 2026-05-17:

```text
Project-compatible HLoc init caches:
  result/result/feature_extract/pose_init_exports/*_hloc_superpoint_superglue_test_20260517.npz
Summary:
  result/result/feature_extract/pose_init_exports/cambridge_hloc_superpoint_superglue_test_20260517_summary.json

OldHospital HLoc render-at-init RGB LoFTR:
  init 144.0mm / 0.306deg median, 255.0mm mean, 86.81% at 50cm/5deg
  refined/fallback cache 167.4mm / 0.264deg median, 235.0mm mean,
    91.76% at 50cm/5deg
  PnP successes 182/182, median inliers 2791
  status: mean improves but median translation worsens; not a stable
    public-init improvement

ShopFacade HLoc render-at-init RGB LoFTR:
  init 41.6mm / 0.204deg median, 63.5mm mean, 99.03% at 50cm/5deg
  refined/fallback cache 56.7mm median, 7.10m mean, 71.84% at 50cm/5deg
  successful-refinement subset 24.0m / 56.9deg median, PnP successes 28/103
  status: catastrophic divergence from a strong public initializer

Query-mode control:
  ShopFacade HLoc smoke16 ref_mode=query -> 0.016mm / 0.000deg median
  OldHospital HLoc smoke16 ref_mode=query -> 0.77mm / 0.000deg median
  interpretation: depth+PnP and HLoc pose conversion are sound; the blocker is
    rendered RGB appearance matching.

POFD Stage4 on HLoc init caches:
  configs:
    feature_extract/configs/pofd_stage4_oldhospital_hloc_superpoint_superglue_full182_eval.yaml
    feature_extract/configs/pofd_stage4_shopfacade_hloc_superpoint_superglue_full103_eval.yaml
  OldHospital full182:
    result/result/feature_extract/eval_pofd_stage4_oldhospital_hloc_superpoint_superglue_full182_20260517/stage4_eval_summary.json
    init/pred cost 0.2558m -> 0.2558m, trans 0.2550m -> 0.2550m,
    rot 0.4108deg -> 0.4108deg, solver_success 0.0, accepted 0.0,
    render_count 3.0, success 5/10/25/50cm unchanged at
    0.082/0.313/0.665/0.868.
  ShopFacade full103:
    result/result/feature_extract/eval_pofd_stage4_shopfacade_hloc_superpoint_superglue_full103_20260517/stage4_eval_summary.json
    init/pred cost 0.0641m -> 0.0641m, trans 0.0635m -> 0.0635m,
    rot 0.3092deg -> 0.3092deg, solver_success 0.0, accepted 0.0,
    render_count 3.0, success 5/10/25/50cm unchanged at
    0.592/0.903/0.961/0.990.
  status: the current POFD Stage4 updater does not improve public HLoc inits.
```

Protocol consequence: a paper-grade claim must report three isolated columns:
`real_init`, `POFD single-render refinement`, and `oracle/cache reranker`.
The current local best full182 real-init numbers come from NetVLAD plus
render-LoFTR/PnP candidate generation, so they are valid as an initializer or
baseline but cannot be reported as the POFD single-render contribution.
The Stage4 POFD path must beat or clearly improve that initializer without
reintroducing multi-candidate inference before claiming SOTA.
The HLoc bridge has produced pose tables for all five official retriangulated
Cambridge model scenes. This is initializer/baseline evidence rather than a
POFD refinement gain. The rendered-RGB refiner does not improve the HLoc
initializer reliably, and POFD Stage4 exact no-ops on HLoc inits for OldHospital
and ShopFacade. The current usable additional-scene evidence remains ShopFacade
external-init, HLoc baseline, and POFD no-op diagnostics rather than a
promotable POFD cross-scene result.

Interpretation update:

```text
The all-candidate Stage4 pair-flow margin creates a real but small continuous
CPR signal on q25/q50. The same relaxation is unsafe for q10, and longer mixed
training over-accepts q25. The PMED selected hard-negative branch improves some
internal pair-flow metrics but does not improve q50 CPR or q50 val128 ranking.
Sparse top-k score-map aggregation also fails: it discards too much supporting
evidence and worsens both the Stage3a baseline and the best Stage4 all-margin
checkpoint on q50 val128.
The new failure dump shows that remaining q50 errors are not random noise:
every dumped wrong selection assigns a higher score to the wrong candidate than
to the oracle candidate. Stage4 all-margin slightly lowers the average failure
cost but increases the wrong-vs-oracle score margin, so this branch is making
some wrong local peaks more confident rather than fixing candidate ranking.
The exact positive-only GeoNCE interpretation of PMED was also tested. It does
make the implementation match the recommendation-file wording more closely:
hard negatives no longer receive offset CE, only cross-candidate margin. In
practice it halves the supervised flow points and worsens both q50 and mixed
20-step candidate ranking; increasing the cross-candidate margin weight from
0.25 to 1.0 does not recover the loss. The isolated recommendation-matrix
smokes confirm the same conclusion: D1 cost-shaped listwise alone drops q50
top1 to 0.875 and D2 positive-only GeoNCE alone reproduces the 0.153m q50
plateau rather than the old cross-candidate 0.129m/top1=1.0 result. D3
listwise+GeoNCE worsens q50 again. On mixed q10/q25/q50, D2 can recover
top1=1.0 but still trails old cross-candidate PMED and D3 again degrades. Treat
this as a no-go for the current pair-flow selected-hard-negative branch.
The full q50 val128 matrix is below the recommendation-file promote line before
score calibration: old cross-candidate PMED r12 reaches only 0.220m/top1
0.727/oracle gap 0.091, while exact D3 r12 is 0.226m/top1 0.711/oracle gap
0.096. D4 r16 eval-only does not help either. D5 is mixed for exact D3: q10
passes cost/success, but q25 is 0.111m with succ@10cm/5deg 0.758.
Failure/all-dump analysis shows the missing term is the opposite of the earlier
delta penalty: selected candidates under-correct relative to the cached init.
An explicit translation correction reward of 2.0/m lifts old cross-candidate
PMED to 0.160m/top1 0.898/oracle gap 0.030 on q50 val128, while q25 and q10
remain strong. This is the first controlled-cache candidate selection result
that clears the q50 cost/top1/success/oracle-gap gates and fixes q25 regression.
It still does not clear the spearman >=0.58 ranking-calibration target on q50.
An offline calibrated score-head probe confirms the trade-off: without importing
the render-LoFTR/PnP cache metrics, simple score+delta ridge models either keep
good selection with weak rank calibration or approach the rank target while
missing the q50 cost/top1/success gates. Importing cache PnP metrics recovers
the q50 oracle under even/odd cross-validation, but that is a multi-render
LoFTR/PnP selector baseline rather than a POFD single-render result.
The POFD-only geometry-neighborhood aggregation branch reaches the same
conclusion from a different angle: selection-tuned smoothing remains around
0.214m/top1 0.766, while rank-tuned smoothing gets a high Spearman proxy only by
regressing mean cost to 0.267m.
The render-once virtual trust-region fallback is implemented and validates the
paper-motivated "one render + K projective samples" path, but the current
feature-energy surface is too flat/noisy. Conservative thresholds become a
no-op, while accepting weak virtual peaks hurts q50 and q10. Keep it as an
ablation-capable module, not as a promoted main result.
The first real-init smoke is also not promotable: the NetVLAD/render-LoFTR
initializer is already near 7cm on the sampled rows, Stage4 correspondence
solves accept no updates, and virtual trust-region updates are harmful.
Lowering the solver min-point gate proves the default no-op is not merely an
overly strict threshold: once updates are allowed, the first update is already
destructive despite low reprojection residuals. The corrected diagnostics show
the dominant issue is not a 50-degree rotation proposal; that was a unit bug in
the logging. The harmful default proposal is about 37cm / 0.96deg with weak
positive correction cosine, and the SE(3) per-iteration translation clamp does
not directly bound camera-center displacement. The new camera-center/rotation
step gate restores safe no-op behavior and tiny solver clamps give only
1-2mm mean gain on val128, far below the 20% real-init improvement target.
Stage4 still needs an update-direction or pose-improvement confidence test
before any real-init deploy claim.
The proposal virtual acceptance gate is not that missing confidence test. On
top50 smoke32 it makes the harmful virtual warm-start much safer, but one-step
thresholds still regress the initializer by 1.7-2.7mm and three-step variants
remain negative even when the gate suppresses most later updates. Repeated
ungated updates also reduce 10cm success from 0.781 to 0.656, confirming that
the solver direction error accumulates faster than any refinement signal. The
val128 dump sweep further shows that confidence tuning cannot rescue the current
proposal generator: oracle accepting only beneficial current updates still gives
less than 8mm gain against a 37cm mean initializer.
Real-init pair-flow fine-tuning on the deploy-style NetVLAD/render-LoFTR top50
cache improves the proposal quality but still does not reach the paper gate.
The best single-render w2 checkpoint gives 0.3700m -> 0.3662m on val128
(+3.79mm), with better valid-match coverage (0.514 vs 0.336) and lower
reprojection error, but no 10cm/5deg success gain. A two-render/two-update
SE(3) run gives 0.3700m -> 0.3647m (+5.36mm) and
raises 10cm/5deg from 0.336 to 0.352 and 25cm/10deg from 0.617 to 0.641.
However it also worsens average rotation from 0.53deg to 0.97deg and drops
5cm/2deg from 0.094 to 0.062, so it is not a deployable continuous-refinement
result. Three updates raise mean gain to +5.87mm but further degrade rotation
to 1.17deg and lose the 10cm/5deg gain. A strict reprojection/rotation accept
gate reduces the two-update gain to +4.28mm without solving the rotation
regression. Fixing the solver to translation-only confirms that rotation
freedom is one failure amplifier, but it still does not create a paper-grade
refiner: the best trans-only run is two 0.5cm-clamped updates at
0.3700m -> 0.3640m (+6.00mm, 1.6% relative), with rotation held at 0.53deg,
but 5cm/2deg drops to 0.086, 10cm/5deg drops to 0.320, and 50cm/10deg drops to
0.773. Three translation-only updates fall back to +4.03mm, so repeated
updates are still not self-correcting. Full182 confirms the same no-go:
0.5cm-clamped translation-only iter1/iter2/iter3 improve only
+3.96mm/+5.61mm/+4.19mm on a 0.4403m initializer. Iter1 is safest
(5cm/10cm/50cm unchanged or improved), iter2 has the best mean cost but
regresses 5cm/2deg, 10cm/5deg, and 50cm/10deg success, and iter3 continues the
success regression while losing mean-cost gain. The 0.25cm iter2 variant is
safer but only +3.72mm. The full182 generated-update oracle reaches +9.19mm
for one 0.5cm step, +16.49mm for two 0.5cm steps, +22.46mm for three 0.5cm
steps, and +9.10mm for the 0.25cm two-step setting, so even perfect
accept/reject over the current proposals would not approach the required 20%
improvement. The median audit is also negative for paper claims: the best
full182 single-render median translation is about 21.7cm, while local
multi-render render-LoFTR/PnP baselines are 16.1-16.7cm and HLoc's public
Hospital reference is 15cm. All-candidate top50/top16 pair-flow
training was also probed, but the bottleneck moves to
`pair_matcher_local_candidate_score_maps`; interrupted
top50/top16 runs logged only 8/4 steps before being stopped, and their latest
target offsets were still near-zero (0.006/0.017px). The branch is recorded as
computationally impractical and not materially different enough in the current
scoring path rather than a promotable result.

POFD-DenseFlow was implemented as the next single-render proposal source:
render one current pose, predict rendered-centered dense query flow/confidence
from POFD query/render features and render geometry, and reuse the existing
robust 2D-3D Stage4 solver. Two 80-step mixed q10/q25/q50 smoke trainings ran
in parallel on the two 3090 GPUs (`pofd_denseflow_mixed_s80_seed0_20260517`
and `pofd_denseflow_mixed_s80_seed1_20260517`). The head is wired into
training/checkpointing and the `stage4_match_source=denseflow` eval path, but
the first val128 real-init result is a no-go: the learned flow is still near
zero, match coverage is only 3.1%, and the solver accepts every tiny update.

| setting | init | final | gain | success@5cm/2deg | success@10cm/5deg | success@25cm/10deg | success@50cm/10deg | accept | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DenseFlow seed0 val128 | 0.3700m | 0.3767m | -6.71mm | 0.094 -> 0.078 | 0.336 -> 0.297 | 0.617 -> 0.633 | 0.781 -> 0.781 | 1.000 | stop |
| DenseFlow seed1 val128 | 0.3700m | 0.3768m | -6.75mm | 0.094 -> 0.078 | 0.336 -> 0.297 | 0.617 -> 0.633 | 0.781 -> 0.781 | 1.000 | stop |

This fails the dense-flow promote/continue gate and should not be expanded to
full182 in its current form. The implementation remains useful for ablation and
future proposal-source work, but the trained dense head is not yet producing a
usable correction direction.

As a non-learned single-render continuous baseline, the existing
`pose_refine/tools/eval_featuremetric_real_init.py` path was reconnected to
the locally available v68 64d query feature export through
`pose_refine/configs/featuremetric_realinit_v68_single_render_eval.yaml`. The
original pose-refine matcher config was not runnable on this machine because
its 96d `features_query_adaptive_v7_locaware_matcher_v1` export is absent.
The repaired diagnostic uses the real NetVLAD+render-LoFTR top50 quality init
cache, renders once per current pose, and runs featuremetric Gauss-Newton on
the DCFF/RADIO feature residual. Two 64-sample smokes ran in parallel on the
two 3090 GPUs. Both are negative:

| setting | init median | refined median | recall@1deg/50mm | recall@1deg/100mm | improved | status |
|---|---:|---:|---:|---:|---:|---|
| v68 FM-GN 10iter damping 0.01 | 0.245deg / 104.5mm | 0.232deg / 124.9mm | 6.2% | 32.8% | 9/64 | stop |
| v68 FM-GN 10iter damping 0.05 | 0.245deg / 104.5mm | 0.230deg / 128.2mm | 7.8% | 32.8% | 9/64 | stop |

This rejects direct featuremetric GN over the current DCFF/RADIO feature
residual as a paper path. It has the right single-render continuous structure,
but the residual is not aligned with the real-init correction direction.

A stronger external-matcher single-render baseline was then added to
`pose_refine/tools/eval_render_loftr_refine.py`: the lightweight render-at-init
LoFTR+PnP diagnostic now accepts the same real-init pose cache used by Stage4.
This uses one RGB/depth render at the current NetVLAD+render-LoFTR pose, runs
LoFTR between query and the rendered image, and solves a fresh PnP from the
render depth. `kornia==0.6.12` was added to `requirements.txt` because the
repository's LoFTR code path requires Kornia but it was not listed in the
environment. This baseline is not a POFD contribution, but it is the right
SOTA-style comparator and a useful teacher/proposal source.

| setting | samples | init median | refined median | mean trans | improved | status |
|---|---:|---:|---:|---:|---:|---|
| render-at-init LoFTR conf0.3/r4 s64 | 64 | 0.245deg / 104.5mm | 0.112deg / 85.5mm | 113.8mm | 40/64 | continue |
| render-at-init LoFTR conf0.2/r8 s64 | 64 | 0.245deg / 104.5mm | 0.122deg / 91.5mm | 117.6mm | 40/64 | weaker |
| render-at-init LoFTR conf0.3/r4 full182 | 182 | 0.413deg / 223.3mm | 0.248deg / 156.3mm | 273.4mm | 133/182 | baseline-positive |
| render-at-init LoFTR conf0.2/r8 full182 | 182 | 0.413deg / 223.3mm | 0.245deg / 177.7mm | 296.2mm | 122/182 | weaker |

The strict setting gives a 67.0mm median gain on full182 (about 30.0% relative
to the same real initializer), beating the Stage4 POFD solver proposals by a
large margin and landing in the same median-translation range as the local
multi-render render-LoFTR/PnP baselines. The result strengthens the paper
protocol but also raises the bar: a journal-grade POFD method must either
distill this single-render LoFTR correction signal into its own feature/matcher
or explicitly present LoFTR+PnP as an external baseline rather than the core
contribution.

The best strict full182 run was also exported as a pose-init cache for the next
teacher/proposal stage:
`result/result/feature_extract/pose_init_exports/oldhospital_renderatinit_loftr_refined_top50qf_full182_conf03_r4_20260517.npz`.
It contains 182 refined poses, `refine_success` for all samples, and LoFTR
inlier/raw-match counts.

That refined-pose cache was then fed back into POFD Stage4 as a one-candidate
real-init cache to test whether the current learned updater adds value on top
of the strongest available single-render external initialization. A small
schema mismatch was fixed first: legacy refined-pose caches only stored
`query_image_names`, `query_image_stems`, `pose_inits`, and `init_sources`,
while Stage4's retrieval-cache loader expected `query_img_ids` and retrieval
metadata. The loader now supplies deterministic fallback fields for minimal
pose-init caches, and future render-LoFTR exports write single-candidate
retrieval-compatible fields directly.

| setting | init mean cost | final mean cost | gain | 5cm/2deg | 10cm/5deg | 25cm/10deg | 50cm/10deg | accept | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| POFD top50/w2 on LoFTR refined full182 | 0.2740m | 0.4197m | -145.7mm | 0.099 -> 0.049 | 0.335 -> 0.159 | 0.637 -> 0.385 | 0.868 -> 0.720 | 0.390 | stop |
| POFD all-margin q50 on LoFTR refined full182 | 0.2740m | 0.2740m | 0.0mm | 0.099 -> 0.099 | 0.335 -> 0.335 | 0.637 -> 0.637 | 0.868 -> 0.868 | 0.000 | safe no-op |

This confirms that the external LoFTR refined cache is a useful teacher and
strong baseline, but not a solved POFD contribution. The real-init top50/w2
updater over-trusts its learned correspondences and damages the strong init;
the all-margin q50 checkpoint is safer only because it refuses to update. The
next POFD-owned branch must learn from the LoFTR correction signal or change
the paper contribution boundary.

A direct render-at-init LoFTR teacher-flow distillation branch was then added
to test that next step. The LoFTR diagnostic can now export
`TeacherCorrespondenceStore`-compatible per-query `.npz` files while refining
real-init caches. The train split was exported in two GPU shards and merged
into:
`result/result/feature_extract/teacher_corr/oldhospital_train_renderatinit_loftr_top50qf_full895_conf03_r4_20260517`.
It contains 895 per-query correspondence files. A full test export was also
generated at:
`result/result/feature_extract/teacher_corr/oldhospital_test_renderatinit_loftr_top50qf_full182_conf03_r4_20260517`
with 182 files. The corresponding full test LoFTR diagnostic reaches
223.3mm/0.413deg init median to 173.1mm/0.264deg final median in this
correspondence-export run; it is still an external teacher/baseline, not a
POFD result.

The first implementation supervised the Stage4 query-centered pair matcher
with teacher `query_xy -> render_xy` offsets. A critical quantization issue
appeared immediately: at the Stage3a 34x60 score grid, only 0.46% of sampled
teacher offsets round to a nonzero pixel. A high-resolution 136x240 companion
config was therefore added, where about 26.3% of train teacher offsets are
nonzero after rounding. The loss also gained a `min_target_offset_px` filter so
the branch can train only on correction-carrying teacher matches rather than
reinforcing center matches.

| setting | teacher-flow signal | Stage4 init cost | Stage4 final cost | gain | 5cm/2deg | 10cm/5deg | status |
|---|---:|---:|---:|---:|---:|---:|---|
| teacher-flow 34x60 smoke | target offset rounds to 0.0px | diagnostic only | diagnostic only | n/a | n/a | n/a | rejected |
| teacher-flow 136x240 w1 s20 argmax | val nonzero frac 0.028, acc 0.022 | 0.0820m | 0.0934m | -11.4mm | 0.125 -> 0.031 | 0.781 -> 0.750 | stop |
| teacher-flow 136x240 w2 s20 argmax | val nonzero frac 0.028, acc 0.026 | 0.0820m | 0.0933m | -11.3mm | 0.125 -> 0.031 | 0.781 -> 0.750 | stop |
| teacher-flow 136x240 w1 s20 expected | soft expected-offset proposal | 0.0820m | 0.0849m | -2.9mm | 0.125 -> 0.031 | 0.781 -> 0.844 | stop |
| teacher-flow 136x240 w2 s20 expected | soft expected-offset proposal | 0.0820m | 0.0847m | -2.7mm | 0.125 -> 0.000 | 0.781 -> 0.844 | stop |

This branch distills the external LoFTR signal into the POFD-owned matcher in
the most literal way, but the current integer-offset Stage4 proposal path is
still not the right target: at deploy-relevant grid scales most teacher
corrections are subpixel, while the higher-resolution integer proposal
over-updates and has negative proposed-update correction cosine. Replacing
argmax with the matcher expected offset reduces the damage from about 11mm to
about 3mm on the smoke split, but it still regresses mean cost and small-error
success. The integer-offset teacher-flow branch is therefore a no-go.

A continuous/subpixel teacher-flow branch was then added so the teacher target
is not rounded away. The loss now supports a subpixel L1/SmoothL1 component
over the local-grid expected offset, separate CE and subpixel weights, and a
continuous `min_target_offset_px` filter. Two 40-step probes do learn the sparse
teacher signal to about 0.382px EPE, but using the resulting dense expected
offset as a Stage4 proposal is destructive because the field has extremely low
confidence and large off-target full-grid offsets.

| setting | train signal | Stage4 init cost | Stage4 final cost | gain | confidence | offset | status |
|---|---:|---:|---:|---:|---:|---:|---|
| subpixel w5 s40 expected | val EPE 0.3819px | 0.0820m | 0.3819m | -299.8mm | 3.95e-05 | 4.86px | stop |
| subpixel w10 s40 expected | val EPE 0.3818px | 0.0820m | 0.3748m | -292.8mm | 3.95e-05 | 4.86px | stop |
| subpixel w10 s40 expected, conf >= 0.001 | same checkpoint | 0.0820m | 0.0820m | 0.0mm | gated out | valid frac 0.0003 | safe no-op |

The confidence threshold prevents damage but produces no refinement signal.
This closes the literal LoFTR teacher-flow path in both integer and current
subpixel forms. Future POFD-owned work needs a different proposal
representation or objective, not a longer run of this same dense expected-flow
head.

The existing PoseEnergy/residual-update branch is also not an untried escape
hatch. Code audit confirms that `PoseEnergyNet` already has energy,
confidence, factorized translation/rotation heads, residual deltas, direction
losses, correction-cosine losses, and residual-update metrics. The historical
small-basin Stage2 pair-heatmap run validates the architecture only in the easy
25cm setting (eval about 0.065m, Spearman 0.71, top1 0.969, oracle gap near
zero). The q50 pair-heatmap PoseEnergy runs fail on the actual target regime:
best q50 eval is about 0.484-0.512m with negative Spearman and only
0.00-0.06 top1 on the train256 joint/factorized variants, while the earlier
selector run is 0.499m with Spearman -0.041. Residual-update gain is zero in
these promoted logs. Do not spend more GPU on the old PoseEnergy selector or
residual head unless the design changes the proposal signal itself; it would
repeat a q50-negative branch.

A stricter single-render PoseEnergy residual pivot was also tested to avoid
the old q50 selector target: use the real top1 render init, disable pair-matcher
and teacher-correlation losses, zero-initialize the residual head, and train
only a direct pose-energy residual update from local correlation. This finds a
small smoke-split positive result, but the explicit full182 evaluation after
removing the inherited `max_val_samples: 32` cap is only sub-millimeter and
does not move success rates.

| setting | split | init cost | residual cost | gain | status |
|---|---:|---:|---:|---:|---|
| nonzero residual scale1 s40 | smoke32 | 0.0820m | 0.2126m | -130.6mm | rejected |
| nonzero residual scale0.5 s40 | smoke32 | 0.0820m | 0.1232m | -41.2mm | rejected |
| zero-init residual scale1 s40 | smoke32 step20 | 0.0820m | 0.0764m | +5.6mm | diagnostic |
| zero-init residual scale0.5 s40 | smoke32 step20 | 0.0820m | 0.0782m | +3.8mm | diagnostic |
| zero-init residual scale1 s40 | full182 | 0.4403m | 0.4399m | +0.47mm | stop |
| zero-init residual scale0.5 s40 | full182 | 0.4403m | 0.4398m | +0.50mm | stop |

The residual pivot is a useful sanity check because it proves the residual
path can be kept no-op safe with zero init, but the full182 gain is far below
the 20% real-init target and leaves 5cm/10cm/25cm/50cm success unchanged. It
should not be promoted or expanded without a materially stronger proposal
signal.

The expert-file stop/pivot rule then points to a GS-CPR-style external
correspondence refiner plus POFD uncertainty/reranking. The current workspace
does not contain a native MASt3R/DUSt3R entrypoint, so the practical external
correspondence source is the existing render-at-init LoFTR+PnP path. Two cache
tools were added for this pivot:

```text
feature_retrieval/tools/gate_refined_init_cache.py
feature_retrieval/tools/append_refined_init_candidate.py
```

The first tool quality-gates an external refined pose cache against the
original real-init cache. On the strict full182 render-at-init LoFTR refined
cache, inlier/raw-match threshold sweeps do not improve over accepting all
refined poses: accept-all remains best at 156.3mm median / 273.4mm mean
translation, while stricter inlier or inlier-ratio gates discard more good
updates than bad ones. Simple LoFTR quality gating is therefore not enough.
The same tool now also supports a no-GT pose-step gate. Using the one-pass to
iter2 camera-center step as the gate signal and exporting the iter3 pose only
when that step is >=0.20m selects 35/182 iter3 poses and improves the external
cache mean to 256.2mm while keeping median translation near the one-pass cache
at 159.6mm. Feeding this step-gated cache back through the all-margin q50
Stage4 evaluator gives an exact no-op: 0.2569m init cost, 0.2569m pred cost,
0.0 accepted updates, and unchanged 5/10/25/50cm success. This is a better
external LoFTR baseline/teacher, not evidence that POFD has learned the update.

As a first POFD-owned uncertainty check for the external correspondence branch,
`pose_refine/tools/score_pose_cache_feature_consistency.py` now scores pose
caches by normalized query/render DCFF feature residual, and
`gate_refined_init_cache.py` supports lower-is-better score-delta gating with
an optional pose-step condition. The full182 one-pass and iter3 caches were
scored in parallel on two GPUs:

```text
result/result/feature_extract/pose_init_exports/oldhospital_renderatinit_loftr_refined_top50qf_full182_conf03_r4_feature_score_20260517.npz
result/result/feature_extract/pose_init_exports/oldhospital_renderatinit_loftr_refined_top50qf_full182_conf03_r4_iter3_feature_score_20260517.npz
```

The raw DCFF residual signal is weak: one-pass and iter3 have nearly identical
mean residuals (0.71954 vs 0.71946), the residual-improvement/translation-gain
correlation is only 0.009, and accepting iter3 whenever the residual is lower
selects 97/182 rows but worsens the one-pass median (164.6mm vs 156.3mm). A
conservative residual-only gate can keep median at 154.6mm and mean at
264.3mm, but it is still weaker than the pose-step gate.

The useful deploy-style combination is: keep one-pass by default, switch to
iter3 only when the one-pass->iter2 step is >=0.20m and the iter3 DCFF residual
is no worse than one-pass by more than 0.001. This selects 27/182 iter3 poses
and exports:

```text
result/result/feature_extract/pose_init_exports/oldhospital_renderatinit_loftr_refined_top50qf_full182_conf03_r4_iter3_step020_featuredelta_ge_m001_20260517.npz
feature_extract/configs/pofd_stage4_single_render_realinit_renderloftr_stepfeature_full182_eval.yaml
```

| cache | selected iter3 | median trans | mean trans | median rot | mean rot | succ@5/10/25/50cm | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| one-pass render-at-init | 0/182 | 156.3mm | 273.4mm | 0.248deg | 0.387deg | 0.099 / 0.335 / 0.637 / 0.868 | strongest median external cache |
| iter3 pose-step gate | 35/182 | 159.6mm | 256.2mm | 0.225deg | 0.359deg | 0.104 / 0.346 / 0.654 / 0.896 | best prior external mean |
| iter3 step+feature gate | 27/182 | 156.3mm | 255.7mm | 0.226deg | 0.358deg | 0.104 / 0.346 / 0.654 / 0.890 | tiny mean gain, preserves one-pass median |

Stage4 all-margin q50 on the step+feature cache is again an exact no-op:
`stage4_init_cost_m = stage4_pred_cost_m = 0.2562968`, accepted updates 0.0,
and 5/10/25/50cm success 0.104/0.346/0.654/0.890. The DCFF residual gate is a
small external-cache uncertainty improvement, but its weak correlation and
post-hoc threshold sensitivity make it insufficient as a top-journal POFD
contribution.

The second tool appends the render-at-init LoFTR refined pose as candidate 51
behind the original NetVLAD+render-LoFTR top50 PnP cache, producing:
`result/result/feature_extract/pose_init_exports/oldhospital_netvlad_renderloftr_top50_plus_renderatinit_loftr_refined_full182_20260517.npz`.
This gives a clean reranking diagnostic without claiming a deployable POFD
method. Full182 pose-cache analysis shows that the external refined candidate
adds real upper-bound value, but it is not always the oracle:

| split | active init median/mean | refined median/mean | oracle top50 median/mean | oracle top50+refined median/mean | refined top1 | refined top8 |
|---|---:|---:|---:|---:|---:|---:|
| first64 | 104.5/146.5mm | 85.5/113.8mm | 37.3/74.4mm | 33.4/63.6mm | 16/64 | 39/64 |
| full182 | 223.3/439.2mm | 156.3/273.4mm | 78.9/115.6mm | 74.3/110.1mm | 24/182 | 109/182 |

The current Stage3a/Stage4 POFD scorer cannot exploit that candidate set.
After fixing the dataset-level `pose_candidate_topk` cap to include all 51
candidates, the 64-sample POFD reranking smoke is negative:

| setting | pred cost | oracle cost | top1 | spearman | pred trans | refined selected | refined oracle | worse than init | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| POFD scorer, top50+refined s64 | 0.2228m | 0.0638m | 0.094 | 0.623 | 222.2mm | 4/64 | 17/64 | 38/64 | stop |

The appended external proposal improves the oracle, but current POFD scores
over-select wrong top50 candidates and often do worse than the original init.
This validates the pivot target: the next promotable branch must train a new
POFD uncertainty/reranking head around external correspondence proposals, not
reuse the old pair-matcher score as-is.

The top51 training probe then hit the expected memory wall in the pair-matcher
local score-map path on 24GB GPUs. To keep the old top16 memory footprint, the
append tool now supports `--max_base_candidates`, and compact caches were
exported with the first 15 NetVLAD+render-LoFTR candidates plus the appended
render-at-init LoFTR refined pose. This makes the external proposal trainable
without changing the scorer architecture.

Compact top15+refined analysis confirms that the refined proposal is useful
but the old scorer still leaves a large oracle gap:

| setting | split | pred cost | oracle cost | top1 | spearman | refined selected | refined oracle | worse than init | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| POFD scorer, top15+refined | s64 | 0.1492m | 0.0686m | 0.203 | 0.415 | 12/64 | 20/64 | 22/64 | baseline |
| rank rerank s20 | s64 | 0.1562m | 0.0686m | 0.234 | 0.417 | 12/64 | 20/64 | 22/64 | rejected |
| quality rerank s20 | s64 | 0.1459m | 0.0686m | 0.219 | 0.426 | 12/64 | 20/64 | 22/64 | weak positive |
| quality-prior 0.05 | s64 | 0.1489m | 0.0686m | 0.141 | 0.326 | 8/64 | 20/64 | 13/64 | fail-safe only |
| identity fallback 0.02 | s64 | 0.1540m | 0.0686m | 0.141 | 0.415 | 9/64 | 20/64 | 23/64 | rejected |
| quality rerank s80 | s64 | 0.1565m | 0.0686m | 0.219 | 0.389 | n/a | n/a | n/a | overfit/rejected |
| improvement rerank s80 | s64 | 0.1519m | 0.0686m | 0.219 | 0.404 | n/a | n/a | n/a | rejected |

The weak s20 quality reranker does not scale to full182 in a promotable way:
full182 baseline top15+refined reduces the active init mean cost from 0.4403m
to 0.3172m, and the quality-s20 checkpoint reaches only 0.3136m. The full182
oracle remains 0.1337m, refined is oracle on 38/182 rows, and the reranker
still selects poses worse than init on 59/182 rows. This is a useful diagnostic
and a controlled-cache baseline, but not a top-journal main result.

The no-GT pose-step-gated external LoFTR pose was then appended as the compact
candidate 16, after fixing `gate_refined_init_cache_by_pose_step` to preserve
the selected refined pose's `refine_success`, `refine_num_inliers`, and
`refine_num_raw_matches` arrays for downstream quality features. This produced:

```text
result/result/feature_extract/pose_init_exports/oldhospital_netvlad_renderloftr_top15_plus_renderatinit_loftr_stepgated_full182_20260517.npz
feature_extract/configs/pofd_stage4_realinit_top15_plus_renderatinit_stepgated_full182_eval.yaml
```

Compact candidate analysis:

| compact cache | active init median/mean | appended median/mean | oracle median/mean | appended better than init | appended oracle | appended top8 |
|---|---:|---:|---:|---:|---:|---:|
| top15 + one-pass LoFTR | 223.3/439.2mm | 156.3/273.4mm | 93.2/133.3mm | 133/182 | 38/182 | 153/182 |
| top15 + pose-step-gated LoFTR | 223.3/439.2mm | 159.6/256.2mm | 91.0/131.1mm | 130/182 | 39/182 | 159/182 |

Full182 POFD/reranker eval on the step-gated compact cache:

| setting | pred cost | oracle cost | top1 | spearman | pred trans | succ@5/10/25/50 | appended selected | appended oracle | worse than init | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Stage3a scorer, top15+one-pass | 0.3172m | 0.1337m | 0.176 | 0.414 | 0.3163m | 0.110 / 0.286 / 0.577 / 0.813 | 32/182 | 38/182 | 57/182 | old baseline |
| Stage3a scorer, top15+step-gated | 0.3160m | 0.1315m | 0.176 | 0.417 | 0.3152m | 0.110 / 0.286 / 0.577 / 0.813 | 30/182 | 39/182 | 56/182 | tiny gain |
| quality-s20 rerank, top15+one-pass | 0.3136m | 0.1337m | 0.165 | 0.407 | 0.3128m | 0.082 / 0.269 / 0.560 / 0.835 | 31/182 | 38/182 | 59/182 | old best |
| quality-s20 rerank, top15+step-gated | 0.3117m | 0.1315m | 0.159 | 0.409 | 0.3108m | 0.082 / 0.269 / 0.560 / 0.841 | 28/182 | 39/182 | 58/182 | tiny gain, still no-go |

The stronger external candidate marginally improves the compact oracle and the
old scorer/reranker, but the gains are only 1-2mm and the selected pose remains
much worse than the external step-gated pose alone (0.256m mean). Current POFD
scoring still over-selects bad base candidates and cannot convert the external
proposal into a publishable reranking result.

As a first additional-scene prerequisite, Cambridge `ShopFacade` was prepared
outside the OldHospital-only loop. RADIO dual v68 features were exported to
`result/result/feature_extract/features_radio_dual_v68_align_stop640/cambridge_shopfacade`
with 334 frames, 64d `fine_geo` and `coarse_sem` tensors at 120x68, and a
`summary_matrix.pt` of shape `(334, 2560)`. A ShopFacade DCFF training config
plus smoke/pilot configs were added under `feature_field/configs/`. The smoke
run completed 8 iterations from the scene COLMAP point cloud with 231/231 train
and 103/103 test cameras matched to cached features. A 2GPU pilot then exposed
and fixed an old-PyTorch DDP initialization incompatibility
(`init_process_group(device_id=...)`), completed 100 iterations at global batch
10, reached best total loss 3.1616 at iter 100, and ran a 10-frame validation
smoke loss of 3.0027. Smoke and pilot checkpoints are load-addressable through
`feature_field/configs/reconstruction_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68_smoke.yaml`
and
`feature_field/configs/reconstruction_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68_pilot100.yaml`.
A longer 2GPU ShopFacade map run was then launched with per-GPU batch size 5.
It saved a best checkpoint at iter 500 with total loss 1.7991 and a latest
checkpoint at iter 2000 with 10-frame validation smoke loss 2.4987. Continuing
to iter 2434 did not beat the iter-500 best and accumulated 198 guarded
non-finite gradient skips, so the run was stopped. The formal reconstruction
config now points to the iter-500 best checkpoint:
`feature_field/configs/reconstruction_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68.yaml`.
The first ShopFacade initialization/refinement diagnostics are also now
exported. Cached RADIO coarse/student top50 banks are too weak for direct
deployment: top1 median translation is about 3.29m and the top50 spatial
oracle is about 0.89m. Raw NetVLAD is stronger but still coarse, with top1
median/mean 1.40/1.79m, 48.5% recall at 10deg/2m, and a top50 spatial oracle
around 0.55m median. The existing real-image LoFTR + rendered-depth PnP
initializer, run over NetVLAD top5 in two GPU shards, is much stronger:
all 515 candidates solve PnP, the selected cache reaches 84.5mm median /
130.4mm mean and 0.403deg median rotation, and the top5 translation oracle is
43.2mm / 71.4mm. A no-GT reprojection-median selector switches away from the
inlier-max candidate on 62/103 queries and improves the selected mean to
111.1mm while keeping the 84.5mm median.

Render-at-init LoFTR smoke runs on 32 test frames diverge from the weak real
initializers: coarse/student start at 2.60m median and end around 35.5m median
with 11/32 PnP successes, while NetVLAD starts at 1.18m median and ends around
26.6m with only 3/32 successes. A query-mode GT-pose sanity check on the same
formal ShopFacade map succeeds on 32/32 frames and refines 65.3mm synthetic
noise to near zero. However, render-at-init LoFTR still diverges even from the
strong 84.5mm real-image LoFTR+depth PnP cache: iter1 and iter2 both end with
27/103 successful but wrong PnP poses, 28.8m final median translation, and zero
improved samples. The immediate lesson is that ShopFacade rendered RGB is not
a reliable LoFTR reference; real-image matching plus rendered depth is usable,
but POFD has not yet added a second-scene contribution.

A ShopFacade Stage4 eval config was added:
`feature_extract/configs/pofd_stage4_shopfacade_realinit_renderloftrpnp_top5_reprojmedian_full103_eval.yaml`.
The first smoke exposed that 1080px input height breaks the query student's
stride-aligned skip fusion (68 vs 67 feature rows), so the config keeps the
existing 1088x1920 model input while retaining 68x120 feature maps. Full103
Stage4 eval on the reprojection-median external init is an exact no-op:
init/pred cost 0.1120m, init/pred translation 0.1111m, init/pred rotation
0.5386deg, solver success 0.0, accepted updates 0.0, match valid fraction
0.150, reprojection error 8.51px, and 5/10/25/50cm success
0.262/0.631/0.913/0.990. This is not a multi-scene POFD result yet: a
non-divergent POFD-owned refiner/uncertainty module and a public comparison
table are still missing.

ShopFacade real-image LoFTR + rendered-depth teacher correspondences were then
exported to test whether the viable external route can train a POFD-owned
single-render update. The train store
`result/result/feature_extract/teacher_corr/shopfacade_train_netvlad_renderloftrpnp_top5_20260517`
covers 229/231 frames, and the test store
`result/result/feature_extract/teacher_corr/shopfacade_test_netvlad_renderloftrpnp_top5_20260517`
covers 103/103 frames, with 512 correspondences per file. The selected train
poses are 64.9mm median / 220.9mm mean; the selected test poses match the
84.5mm / 130.4mm external init.

Direct teacher-quality training is runnable but not useful: full-val candidate
selection worsens to about 0.167m pred cost versus the 0.131m active selected
external cache. Pair-flow-only probes isolate the offset branch:

| ShopFacade teacher-flow setting | train/eval signal | Stage4 init cost | Stage4 final cost | update gate | status |
| --- | --- | --- | --- | --- | --- |
| 34x60 CE/margin, top1 PM only | eval acc 0.385, nonzero offset frac 0.038 | 0.1120m | 0.5139m | confidence 0.0, accept 0.288 | stop |
| 136x240 subpixel, expected offsets | eval acc 0.361, subpixel EPE 0.850px, nonzero frac 0.440 | 0.1120m | 0.2110m | confidence 0.0, accept 0.201 | stop |
| 136x240 subpixel, confidence 0.001 | same checkpoint | 0.1120m | 0.1259m | accept 0.023 | stop |
| 136x240 subpixel, confidence 0.002 | same checkpoint | 0.1120m | 0.1120m | accept 0.000 | no-op |
| 136x240 subpixel, confidence 0.005/0.010 | same checkpoint | 0.1120m | 0.1120m | accept 0.000 | no-op |

The second scene therefore repeats the OldHospital lesson: the teacher signal
can train local offset metrics, but current Stage4 proposal geometry and
confidence are not a deployable pose-update mechanism. Thresholding only trades
regression for no-op, so this ShopFacade teacher-flow branch is diagnostic-only.

Two render-at-init LoFTR parameter probes were also run to see whether the
external correspondence baseline itself had easy headroom. Lowering LoFTR
confidence to 0.2 with a 4px PnP threshold gives 162.6mm median / 274.5mm mean.
Using 0.3 confidence with an 8px threshold gives 170.1mm / 282.3mm. The
existing strict 0.3 confidence / 4px threshold remains best at 156.3mm /
273.4mm, so simple LoFTR threshold tuning is also exhausted.

After adding the step+feature external cache, Stage4 was debugged at the
proposal/solver/acceptance boundary. The default all-margin q50 Stage4 eval is
an exact no-op because `stage4_min_points=64` marks the solver unsuccessful:
on an 8-sample dump the matcher is not empty (`match_valid_frac` about 0.334,
confidence about 0.471, reprojection about 0.77px), but the solver has only
about 40 inliers. Lowering `stage4_min_points` to 16 proves the gate diagnosis
but exposes the real failure: solver success and acceptance become 1.0, yet
smoke32 worsens from 59.5mm mean cost to 829.3mm, and first-iteration
`update_cost_gain` is negative for all 32 samples. Expected offsets and raw
local correlation do not fix the direction: expected-offset smoke32 worsens
59.5mm -> 372.2mm, while local correlation is mostly rejected and still worsens
to 208.0mm. A guarded full182 check with `min_points=16`, one iteration, and
acceptance clamps of 5cm / 0.5deg accepts only 2.2% of updates and effectively
no-ops: 0.25630m -> 0.25648m mean cost, identical threshold success, and a
negative mean proposed update gain of -0.414m. This closes the current Stage4
proposal geometry as non-promotable rather than merely over-thresholded.

The next external-correspondence pivot tested whether POFD/DCFF feature
residual can act as per-match uncertainty for render-at-init LoFTR. The LoFTR
correspondence payload now preserves `pnp_inlier_mask`, and
`pose_refine/tools/score_loftr_correspondence_feature_consistency.py` scores
all depth-valid LoFTR matches by query/render DCFF cosine residual. Full182
all-match export from the step+feature cache produced 525,462 scored matches:
501,474 PnP inliers and 23,988 outliers. DCFF residual is not an inlier signal:
global lower-residual AUC is 0.484, query-mean AUC is 0.444, and outliers have
slightly lower residual on average (0.7487) than inliers (0.7566). Threshold
sweeps confirm this is not a hidden gate: residual high/low quantiles do not
improve PnP beyond the external cache. LoFTR confidence itself is much stronger
as an inlier classifier (AUC 0.875), but confidence-filtered PnP still does not
beat the existing step+feature external pose cache; best median in the sweep is
about 170.3mm, versus 156.3mm for the current cache. Re-running LoFTR all-match
PnP from the step+feature poses also worsens the median to 171.4mm. Therefore
POFD residual-as-correspondence-filter is diagnostic-only/no-go in its current
form.

Follow-up data-surface fix: the real-image LoFTR+render-depth teacher export
path now also preserves `pnp_inlier_mask` in
`feature_retrieval/render_loftr_pnp_init_export.py` for both inlier-only and
all-depth-valid correspondence payloads. This matters for the GS-CPR-style
pivot because all-depth-valid teacher stores can now carry positive/negative
correspondence labels instead of only capped positive examples. Two-query GPU
smokes verified real artifacts on both ShopFacade and OldHospital, then full
top5 allvalid test stores were exported. ShopFacade has 103 files, 515/515
PnP-success candidates, 204,054 labeled correspondences, 179,740 inliers,
24,314 outliers, and confidence higher-is-inlier AUC 0.621. OldHospital has
182 files, 910/910 PnP-success candidates, 334,559 labeled correspondences,
296,450 inliers, 38,109 outliers, and confidence AUC 0.766. This is
infrastructure for a stronger external-correspondence teacher, not a promoted
POFD result.

The first non-leaky learned reliability probe is also diagnostic-only.
`feature_retrieval/tools/score_correspondence_reliability.py` trains a small
logistic head from LoFTR confidence and query position, and reports confidence
baseline vs model AUC/AP/Brier. Same-scene AUC improves only modestly
(ShopFacade 0.621 -> 0.640, OldHospital 0.766 -> 0.776), cross-scene transfer
is mixed (ShopFacade-trained model drops OldHospital to 0.751; OldHospital-
trained model raises ShopFacade only to 0.626), and Brier calibration worsens.
Therefore raw LoFTR confidence remains the best non-leaky correspondence
quality signal for now; injecting this head into PnP/Stage4 is not justified.
Pose-level filtering confirms the conclusion is only a weak external baseline
improvement, not a main-method result. The new
`feature_retrieval/tools/sweep_correspondence_pnp_filters.py` tool re-solves
PnP after keeping top-scored correspondences. ShopFacade improves from
65.9/108.3mm median/mean to 55.7/103.6mm with confidence top-25%, while the
cross-scene model top-75% gives 59.4/92.1mm and 5deg/250mm success 0.951.
OldHospital improves only slightly: best cross-scene model top-90% reaches
200.2/402.0mm from 206.8/421.8mm and 5deg/250mm success 0.571 from 0.549.
The effect is real but small and scene-dependent.

Promote criteria are not met:
  q50 val128 now clears the cost/top1/success/oracle-gap targets only with an
  explicit cached-candidate correction reward, but spearman remains below 0.58.
  Full182 confirms the same pattern: q50 reward=2 reaches 0.179m/top1 0.874
  and succ@25 0.890, but spearman is only 0.481; q25/q10 are protected, with
  q25 at 0.071m and q10 at 0.028m.
  Stage4 q50 improves only +29mm on a four-sample controlled probe and remains
  around 0.49m final cost from the q50 cache init.
  Real-init continuous refinement is still only a 6.00mm mean gain on val128
  and 5.61mm on full182 at its best checked translation-only two-update
  setting, far below the required 20% gain and with small-error / 10cm / 50cm
  success regressions.

Stop/pivot condition is partially triggered for the PMED selected-hard-negative
variant: internal metrics changed, but exact PMED candidate pred_cost and
Stage4 CPR did not improve. The D1/D2 isolation also rules out simply removing
auxiliary teacher/observability/ranking losses as the missing ingredient. The
D3/D4/D5 matrix rules out promoting the recommendation-file Stage3d branch
as-is. The correction-reward branch is useful as a controlled-cache reranker,
but it is a calibrated selection prior rather than a solved pair-matcher score
surface or continuous-refinement result.
```

Next concrete branch:

```text
Keep the correction reward as the current controlled-cache selection baseline
and keep all-candidate pair-flow margin as an experimental Stage4 support loss.
For paper-grade progress, the next branch must improve rank calibration or
continuous refinement rather than only selection:
  1. do not promote cache-PnP metric calibration as POFD; keep it as a
     multi-render render-LoFTR/PnP baseline/upper diagnostic;
  2. do not promote POFD-only score-neighborhood smoothing; held-out selection
     remains far below the q50 gates, and rank-tuned smoothing sacrifices cost;
  3. do not repeat the old PoseEnergy residual/selector branch or the direct
     zero-init residual pivot without a new proposal source; q50 failed and the
     full182 residual gain is only about 0.5mm;
  4. separately continue real-init continuous refinement only if the update
     direction confidence test beats the current 6mm gain without degrading
     5cm/10cm/50cm success;
  5. do not repeat the current render-at-init LoFTR teacher-flow branch as
     integer or dense expected subpixel offsets; both are now measured no-go.
     Either design a genuinely different POFD-owned proposal target or treat
     LoFTR+PnP as an external baseline/teacher outside POFD's core method.
  6. MASt3R/DUSt3R is now staged from the official repo, but only as a
     preflight. Do not implement it as an ad hoc swap for LoFTR; first design
     the adapter boundary around official MASt3R correspondences, existing
     render-depth/world-position tensors, robust PnP, and POFD uncertainty.
```

MASt3R/DUSt3R preflight added on 2026-05-17:

```text
third_party/mast3r
  MASt3R HEAD: f5209afc300cec36239a7ac992263f36847bbba0
  DUSt3R submodule: 3cc8c88c413bb9e34c41db0e0eef99c2ee010b12
  CroCo submodule: d7de0705845239092414480bd829228723bf20de
  upstream license: CC BY-NC-SA 4.0 plus checkpoint dataset-license notice
current env
  Python 3.8.10, torch 1.13.1+cu116, CUDA visible on 2x RTX 3090
  core/visloc imports pass after installing roma, numpy-quaternion, kapture,
  and kapture-localization
  official checkpoint downloaded; bundled-pair GPU smoke returns 1,047 matches
  Cambridge adapter/conversion is still missing for the HLoc retriangulated
  data layout
```

Latest verification after adding the HLoc cache/refiner evidence and MASt3R
preflight:

```text
pytest third_party/Hierarchical-Localization/tests/test_match_workers_env.py \
  third_party/Hierarchical-Localization/tests/test_pycolmap_compat.py -q
  -> 5 passed
pytest tests/test_sweep_correspondence_pnp_filters.py \
  tests/test_score_correspondence_reliability.py \
  tests/test_render_loftr_pnp_init_export.py \
  tests/test_eval_render_loftr_refine.py \
  tests/test_score_loftr_correspondence_feature_consistency.py -q
  -> 27 passed, 2 warnings
python -m py_compile third_party/Hierarchical-Localization/hloc/match_features.py \
  third_party/Hierarchical-Localization/hloc/triangulation.py \
  data/radio_loc_retrieval_dataset.py feature_retrieval/localization_mainline.py \
  pose_refine/tools/eval_render_loftr_refine.py \
  feature_extract/tools/train_nvs_pose_feature_adapter.py
  -> passed
HLoc Stage4 config/result check
  -> OldHospital 0.255759m -> 0.255759m and ShopFacade
     0.064079m -> 0.064079m, both with solver_success=0.0 and accepted=0.0
git diff --check
  -> passed
HLoc init/refined cache load check
  -> five HLoc init caches plus OldHospital/ShopFacade refined caches load
MASt3R commit/import preflight
  -> MASt3R/DUSt3R/CroCo pinned commits verified; core/visloc imports pass;
     synthetic PnP passes for cv2 and pycolmap; local-checkpoint pair smoke
     passes; no Cambridge adapter/result yet
nvidia-smi + process check
  -> both GPUs idle; no HLoc/Cambridge/pytest/render-refine processes remain
```
