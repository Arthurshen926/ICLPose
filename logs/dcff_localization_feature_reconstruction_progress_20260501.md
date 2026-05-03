# DCFF Feature Reconstruction / Localization Progress - 2026-05-01

## Current Status

- Oracle WLS with GT flow is healthy: from roughly 70-90 mm initial translation error, the backend reaches 0-7 mm on the diagnostic subset.
- Raw teacher-map local correlation is not a valid localization signal for either adaptive v7 or cached PCA64. Both zero-shot corr-WLS variants diverge to hundreds of millimeters or worse.
- Exported adaptive v7 query-student features without the learned matcher also diverge; softmax confidence is close to uniform.
- Loading the learned `local_matcher` prevents divergence, but it mostly predicts near-zero flow. This explains why training logs can show reasonable `corr_wls` while `gain` stays around zero.
- High-resolution 544x960 evaluation confirms the same issue: the matcher remains near-zero-flow even when rendering at the training resolution.

## Changes Implemented

- Added `pose_refine/tools/eval_corr_wls.py` support for loading a feature-extract `local_matcher` from a query-student config/checkpoint.
- Added diagnostic configs for oracle WLS, adaptive teacher/query, cached64, and high-resolution query-student corr-WLS evaluation.
- Added `map_supervision.query_corr_min_flow_px` to ignore near-zero-flow targets in local correlation losses.
- Added `model.warmstart_skip_prefixes` support so experiments can warmstart the query encoder while resetting collapsed submodules such as `local_matcher`.
- Added flow magnitude diagnostics in `local_correlation_joint_losses` for future logs:
  - `map_query_corr_gt_flow_mag_px`
  - `map_query_corr_pred_flow_mag_px`
- Added tests for matcher-aware eval, min-flow masking, warmstart prefix skipping, and flow magnitude logging.

## Experiment Notes

- v4 pose-gain warmstart from v3:
  - Stopped after step 600 trend.
  - No stable positive pose gain; `gain` stayed near zero or negative.
- v5 stronger pose-gain scale1:
  - Stopped early because fine/map-query metrics degraded and gain stayed near zero.
- v6 min-flow + large perturb + reset matcher:
  - Started successfully after disabling high-res map-self corr branch.
  - `corr_acc` recovered from near zero on some batches, showing non-zero-flow supervision is active.
  - WLS remained around 45-60 mm with only occasional 1 mm-level gains by step 210.
  - Stopped to avoid low-value GPU burn.

## Technical Conclusion

The current bottleneck is no longer DCFF geometry or WLS. It is the query-map local correspondence model. Descriptor cosine, teacher anchoring, and residual correlation-logit matching are insufficient for centimeter localization because the system can satisfy much of the training objective by learning a high-confidence center/near-zero flow.

## Next Direction

- Replace or augment the residual `local_matcher` with an explicit depth-aware local flow/confidence head.
- Train that head directly on GT flow, robust confidence, and WLS pose gain at the same high-resolution render scale.
- Keep adaptive teacher as a weak descriptor anchor, but make localization losses primary for the fine branch.
- Use min-flow / curriculum perturbation sampling so the model sees enough non-zero displacement while still retaining a final small-perturb refinement regime.
