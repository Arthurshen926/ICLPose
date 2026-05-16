# 2026-05-16 POFD Reset Mainline

## Core Target

The active mainline is **POFD: Pose-Observable Feature Distillation**.
For the current paper claim we focus on controlled pose refinement, not full
single-image relocalization:

```text
given query image + 3D feature field + approximate T0,
distill RADIO/DCFF features into one localization feature that improves
local 6DoF refinement accuracy and enlarges the convergence basin.
```

The current interpretation follows `ChatGPT-ICLPose.md`: POFD is a
localization-oriented distiller, not a generic RADIO compressor.  The feature
must make the local pose energy more separable and optimizable.

## What Is Out Of Mainline

- Full retrieval and feature-bank global topK are appendices/future work.
- Explicit coarse/fine dual-feature claims are paused.
- Learned topK selector / PoseEnergy reranker is not the mainline; rich selector
  replacement degraded q50 validation and was removed as a negative diagnostic.
- WLS step-size tuning and candidate-gradient map fine-tune remain negative
  diagnostics unless a later controlled experiment reverses the conclusion.
- Fixed-lattice reports are diagnostic only; shuffled/jittered reports are the
  preferred validation protocol.

## Current Canonical Stages

1. `pofd_stage1_singleloc_anchor_q10_q25_shuf.yaml`
   - single localization feature adapter
   - teacher/cross-view anchor
   - pair-matcher correspondence
   - no candidate rank or pose-energy head

2. `pofd_stage2_poseobs_q10_q25_shuf.yaml`
   - adds GT-pose-vs-wrong-pose observability contrast
   - uses GT pose error ranking inside the local candidate bank
   - current best clean POFD small-basin result

3. `pofd_stage2d_nvs_quality_pose_margin_q50_shuf.yaml`
   - current q50 probe
   - keeps useful render-LoFTR/PnP NVS teacher-quality evidence
   - adds candidate-internal pose-observability margin
   - still one localization feature; no coarse/fine claim and no selector head

## New Losses

`L_obs_surrogate` compares the adapted query feature against:

- positive: map feature rendered at GT pose
- negatives: map features rendered at nearby wrong candidate poses

It enforces:

```text
score(query, render(T_gt)) > score(query, render(T_wrong)) + margin
```

This is useful but incomplete: Stage2b/Stage2c showed that improving this side
objective can fail to improve topK candidate ordering on q50.

`L_candidate_obs` is now added for the q50 probe:

```text
score(best pose candidate in bank) > score(hard worse candidates) + margin
```

This optimizes the same score surface used at evaluation time and is closer to
pose-energy landscape shaping.

## Experiment Ledger

| Run | Eval bucket | Best/first eval | Conclusion |
|---|---:|---:|---|
| `pofd_stage2_poseobs_q10q25...` | q10/q25 | pred_cost `0.0678m`, oracle `0.0453m`, top1 `0.775` | Current clean POFD small-basin baseline. |
| `pofd_stage2_poseobs_q50...` | q50 | pred_cost `0.3535m`, oracle `0.1294m` | Pure GT/observability does not solve medium basin. |
| `pofd_stage2b_confidence_q10q25...` | q10/q25 | pred_cost `0.0897m` | Confidence weighting hurt small basin. |
| `pofd_stage2b_confidence_q50...` | q50 | pred_cost `0.3549m` | Confidence weighting did not improve q50. |
| `pofd_stage2c_candidate_margin_q10q25...` | q10/q25 | pred_cost `0.0829m` | Candidate margin alone hurt small basin. |
| `pofd_stage2c_candidate_margin_q50...` | q50 | pred_cost `0.3464m` | Slightly better than Stage2b but still below old q50 best. |
| `pofd_stage2d_q50_teacher_quality_baseline...` | q50 | pred_cost `0.3290m` | Same-condition NVS teacher-quality baseline. |
| `pofd_stage2d_nvs_quality_pose_margin_q50...` | q50 | best step40 pred_cost `0.3160m`, oracle `0.1294m`, top1 `0.583` | NVS quality + candidate pose margin gives small gain over same-condition teacher baseline, but not enough. |
| old `nvs_q50_train256_teacher_paircorr...` | q50 | pred_cost `0.3013m` | NVS teacher-quality signal remains useful; should be reframed, not discarded. |
| `pofd_stage2e_q50_anchor_quality_pose_margin...` | q50 | step50 pred_cost `0.3894m` | Stronger anchor/drift did not fix q50 and hurt ranking. |
| `pofd_stage2e_q50_residual_feature_health...` | q50 | step50 pred_cost `0.3356m` | Residual texture preserves align better but still worsens q50 versus Stage2d/old NVS. |

## Current Interpretation

The latest negative results are useful. They say the bottleneck is not an
extra selector head, score prior, or confidence scalar.  The q50 medium-basin
case still lacks a localization feature whose candidate score surface is
stable across validation scenes.  In practice, NVS teacher-quality supervision
(render-LoFTR/PnP quality) is still the best available signal for q50 because it
encodes actual matchability/solvability, not only pose distance.

Therefore the next mainline is:

```text
single localization feature
+ NVS teacher-quality / correspondence supervision
+ candidate-internal pose margin
+ basin-curve validation
```

But Stage2d shows this is still not sufficient for the target q50 basin.  The
next architecture change should not add another selector head.  It should make
the feature itself more pose-observable, e.g. by training with stronger
multi-view NVS correspondence/warping, hard scene negatives, or a true
feature-gradient/Jacobian observability term instead of only candidate logits.

Stage2e and the Fisher/logdet diagnostic sharpen this conclusion: stronger
teacher anchor, residual texture fusion, and raw feature-gradient logdet can
all look healthier while still failing q50 candidate ordering.  The next
feature objective must improve validation score ordering and hard-negative
separation, not merely cosine, alignment, or Fisher magnitude.

not:

```text
coarse/fine retrieval system
+ trainable topK selector
+ confidence-weight patch on the same weak feature
```

## Success Evidence To Track

- q10/q25 and q50 final candidate selected translation and rotation
- selected-vs-oracle gap on shuffled candidates
- candidate-observability gap, not only GT-vs-wrong observability gap
- basin success rates at `5cm/2deg`, `10cm/5deg`, `25cm/10deg`, `50cm/10deg`
- feature visualizations should not collapse to one color/channel

## Runtime Notes

- Use direct `conda activate iclpose && python -u ...`; `conda run` can hide
  stdout and leave long-running jobs stuck behind pipe buffering.
- `MAX_JOBS=1` avoids gsplat extension compile memory spikes.
- Two renderer trainings can now run after the CUDA extension is compiled, but
  only do this for clearly independent, high-value probes.  Avoid parallel
  config thrashing.
- Cleaned old fine-selector caches and failed selector outputs; see
  `docs/superpowers/plans/2026-05-16-output-cleanup.md`.
- Fixed `train_nvs_pose_feature_adapter.py` config precedence so YAML
  `training.batch_size`, `max_steps`, `eval_every`, and `save_every` are used
  unless explicitly overridden on the CLI.  Stage2e exposed that parser defaults
  were previously overriding the intended short-run protocol.
