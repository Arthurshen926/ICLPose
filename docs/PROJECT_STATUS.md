# Project Status

## Current Direction

The main active thread is now:

`feature framework -> pose-refine consumer`

with emphasis on:

- stable map-side coarse/fine features
- perturbation-aware geometric supervision
- `full_wls` as the simplest pose consumer
- query-side lightweight features that stay compatible with the map side

## What Has Been Verified

- `full_wls_smoke` reached **1196.9 mm** median translation.
- geometry-side init jitter helped flow magnitude, but hurt smoke accuracy.
- removing feature match loss alone did not help.
- feature-side perturb-aware training is directionally useful.
- stable-map geomcorr 2e + colmap-id export reached **1331.4 mm** median translation in smoke.
- the stable-map run with the same pipeline still trails the old best smoke baseline, so more scaling is needed.

## Short-Term Goals

1. Scale the best stable-map geomcorr variant beyond smoke.
2. Keep the map side frozen/stable while improving the query/student feature geometry.
3. Re-run the closed loop export -> pose_refine -> real-init evaluation.
4. Watch `trans_med`, `flow_epe`, and predicted flow magnitude.

## Long-Term Goals

1. Reach a coarse-to-fine localization pipeline with distinct coarse semantic and fine geometric features.
2. Keep both map-side and query-side feature hierarchies aligned.
3. Make the final localization path robust enough for centimeter-level translation and sub-degree rotation.
4. Keep the system deployable on a rented GPU server without manual path surgery.

## Deployment Notes

- Fastest deployment path: clone the repo to `/root/ICLPose-loc`.
- Copy `dataset/` and `output/` checkpoints/features to the new machine.
- Use `environment.cloud.yml` as the base environment file.
- Some smoke configs still contain absolute `/root/ICLPose-loc/...` paths, so either keep the same path or rewrite them before running.
