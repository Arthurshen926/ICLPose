# 2026-05-17 POFD-DenseFlow Single-Render Proposal Source Design

## Purpose

The current expert-file objective is not satisfied by the existing PMED/Stage4
branches. Controlled-cache ranking can improve candidate selection, but the
paper-preferred single-render continuous refinement path improves real-init
full182 by only a few millimeters and still regresses several success metrics.
The generated-update oracle is also far below the required 20% initializer
improvement, so the missing component is the proposal source itself rather than
only acceptance calibration.

This design adds a new single-render proposal source, POFD-DenseFlow, that
predicts rendered-centered dense flow and confidence from one render, then uses
the existing WLS/Stage4 pose-update machinery to produce a camera update.

## Recommendation

Use the pose_refine dense-flow design as the starting point, but integrate it
into the feature_extract Stage4 real-init protocol:

```text
query feature
+ current-pose render feature
+ render depth / position / mask / intrinsics
-> dense flow + confidence
-> WLS / robust pose update
-> deployable accept gates and Stage4 diagnostics
```

This is preferred over continuing the current pair-matcher correspondence path
because the pair-matcher proposal oracle has too little upside. It is also
preferred over a virtual featuremetric optimizer because the existing
render-once score surface is too flat/noisy under q50 real-init perturbations.

## Existing Interfaces

The feature_extract adapter training path already exposes the required data:

- `forward_batch(...)` builds `pose_gt`, `init_pose`, query features, candidate
  renders, render depth, render position, masks, and intrinsics.
- `evaluate_stage4_single_render(...)` already evaluates real-init single-render
  updates and writes per-update diagnostics.
- `robust_pose_update_from_correspondences(...)` already computes deployable
  pose-update diagnostics, accept gates, success metrics, and JSONL dumps.

The pose_refine codebase provides the closest working reference:

- `DepthAwareLocalFlowHead` predicts dense flow and confidence from correlation
  plus depth/observability context.
- `ConcatPoseNet._forward_corr_wls(...)` and `_forward_gru(...)` implement
  rendered-centered flow, confidence, and WLS pose updates.
- `ConcatPoseNet.compute_gt_flow(...)` defines a GT flow target from init pose,
  GT pose, rendered depth, and intrinsics.

## Architecture

Add a small POFD dense-flow head that can run inside the feature_extract Stage4
path without depending on multi-candidate inference:

```text
PofdDenseFlowHead
  inputs:
    query_loc:        [B,C,H,W]
    render_loc:       [B,C,H,W]
    render_depth:     [B,1,H,W] or [B,H,W]
    render_mask:      [B,1,H,W]
    intrinsics:       [B,4]
  internal:
    local correlation volume, rendered-centered
    depth/observability context
    residual flow head
    confidence head
  outputs:
    flow_px:          [B,2,H,W]
    confidence:       [B,1,H,W]
    optional corr/debug maps
```

The first implementation should be a narrow adapter around the existing
pose_refine flow-head pattern rather than a new large network. It should allow
the projected POFD features to remain frozen during the first probe, so the
experiment tests whether a new proposal head can exploit the existing feature
surface before changing the feature extractor again.

## Data Flow

Training uses cached real-init/candidate-bank entries only to obtain `pose_init`
and GT pose buckets. It must not use top-K candidate selection at inference.

For each batch:

1. Select the current render pose from `pose_init`.
2. Render exactly one current-pose feature/depth/position bundle.
3. Project query and render features through the existing POFD adapter.
4. Build rendered-centered local correlation.
5. Predict dense flow and confidence.
6. Convert flow/depth/intrinsics to a pose update.
7. Score the update with GT diagnostics during training/eval only.

Evaluation repeats the same render/update path on val128 and full182. It must
report render count, mean/median pose error, success rates, accepted update
fraction, solver diagnostics, and generated-update oracle diagnostics.

## Losses

Use a staged loss schedule:

```text
L = w_flow * robust_flow_loss
  + w_pose * pose_delta_loss
  + w_gain * beneficial_update_loss
  + w_conf * confidence_calibration_loss
```

The initial smoke should prioritize flow and pose-update direction:

- `robust_flow_loss`: Smooth L1 or Charbonnier on valid GT flow pixels.
- `pose_delta_loss`: translation and rotation update target from init to GT.
- `beneficial_update_loss`: penalize updates that increase cost relative to
  init, with a margin for q50 samples.
- `confidence_calibration_loss`: encourage high confidence on low-flow-error
  valid pixels and low confidence on invalid/outlier pixels.

The first branch should keep q10/q25/q50 mixed training active. q50 is the main
target, but q10/q25 regression protection is mandatory.

## Metrics

Add dense-flow proposal metrics alongside existing Stage4 metrics:

```text
denseflow_flow_epe_px
denseflow_valid_frac
denseflow_conf_mean
denseflow_conf_ece_or_proxy
denseflow_delta_trans_m
denseflow_delta_rot_deg
denseflow_update_cost_gain_m
denseflow_update_correction_cos
denseflow_solver_success
denseflow_reproj_px
denseflow_inlier_count
```

The main report must also keep the existing Stage4 real-init metrics:

```text
init_cost_m, pred_cost_m, gain_m
median trans/rot
success@5cm/2deg
success@10cm/5deg
success@25cm/10deg
success@50cm/10deg
render_count
```

## Experiment Matrix

Run narrow probes before any expensive full run:

| ID | training data | head | eval | promote hint |
|---|---|---|---|---|
| F0 | none | current Stage4 pair matcher | val128/full182 | fixed baseline |
| F1 | q10/q25/q50 mixed | dense flow, frozen adapter | val128 | must beat +6mm by a clear margin |
| F2 | q10/q25/q50 mixed | dense flow + confidence loss | val128 | better success rates, lower harmful updates |
| F3 | q50-emphasized mixed | dense flow + gain loss | val128 | >=20% mean gain or strong partial signal |
| F4 | best F1-F3 | same checkpoint | full182 | no val128-only claim |
| F5 | best F4 | 3 seeds | val128/full182 | stability check |

Only move to multi-seed/full protocol if val128 shows material gain without
small-basin or success-rate regressions.

## Promotion Gates

Promote POFD-DenseFlow only if it satisfies all of the following:

```text
q50 val128:
  mean pred_cost improves current real-init initializer by >=20%
  success@25cm/10deg >= 0.80
  no regression in success@5cm/2deg, @10cm/5deg, or @50cm/10deg

full182:
  confirms the val128 gain direction
  mean cost improves the real-init baseline by >=20%
  best median translation beats the current Stage4 single-render baseline
  no evidence that the result is a val subset artifact

paper protocol:
  keep single-render result separate from cache reranker and multi-render
  compare against local multi-render render-LoFTR/PnP as a separate baseline
  run 3 seeds before claiming stability
  add at least one additional scene before top-journal submission
```

Continue only if val128 gain is clearly above the current millimeter-scale
Stage4 ceiling and dense-flow direction metrics improve. Stop the branch if
three controlled configs remain below 5mm gain or if q50 improves only by
trading away q10/q25 or small-error success.

## Implementation Boundaries

Keep the first implementation isolated:

- Add dense-flow modules and losses without removing existing Stage4 code.
- Reuse pose_refine geometry utilities where practical.
- Keep current pair-matcher Stage4 as a baseline, not as a dependency of the new
  proposal source.
- Do not import render-LoFTR/PnP cache fields into the single-render decision
  path.
- Do not claim SOTA from val128 alone.

## Tests And Verification

Unit tests should cover:

- GT flow target computation under identity and translated poses.
- Intrinsics scaling from query/render feature resolution to solver resolution.
- Dense-flow head output shapes and confidence range.
- Pose-update wrapper no-op behavior for zero flow.
- Stage4 dense-flow eval uses cached `pose_init` in cache mode.

Verification commands:

```text
pytest tests/test_feature_extract_checkpoint_io.py tests/test_nvs_pose_feature_adapter.py -q
python -m py_compile feature_extract/tools/train_nvs_pose_feature_adapter.py
git diff --check
```

GPU experiments should start with val128 smokes and only expand to full182 /
multi-seed after F1-F3 produce a signal worth scaling.

## Open Decision

The branch is ready for implementation planning if the approved direction is:

```text
Implement POFD-DenseFlow as the next single-render proposal source, reusing the
pose_refine dense-flow/WLS design but evaluating through feature_extract Stage4
real-init gates.
```
