# Pose Refine Progress - 2026-05-03

## Current OldHospital Full-Val Best

- Checkpoint: `/root/ICLPose/result/pose_refine/concat_loc_cambridge_oldhospital_processed_adaptive_v7_locaware_matcher_gru_flowhead_aftertrans025_outer3_v1_probe/checkpoints/best.pth`
- Inference config: `pose_refine/configs/concat_loc_cambridge_oldhospital_processed_adaptive_v7_locaware_matcher_gru_flowhead_obs_aftertrans0_outer3_v1_full_eval.yaml`
- Inference setting: `outer_iters=5`, `gru_iters=2`, `pose_update_trans_scale_after_first=0.0`
- Full-val result: `73.2mm / 0.345deg / joint@1deg-50mm=22.0%`
- Result JSON: `/root/ICLPose/result/pose_refine/concat_loc_cambridge_oldhospital_processed_adaptive_v7_locaware_matcher_gru_flowhead_obs_aftertrans0_outer3_v1_full_eval/eval_sweep_results.json`

## What Improved

- Previous stable full-val baseline was about `73.3mm / 0.45deg / 19.8%`.
- `aftertrans025 + outer3` improved rotation to `73.6mm / 0.36deg / 20.9%`.
- `aftertrans0 + outer5` preserved translation while further reducing rotation to `73.2mm / 0.345deg / 22.0%`.
- The improvement comes from using later refine iterations as rotation-only updates. This matches the observed tradeoff: more outer iterations reduce rotation but can drift translation if later translation updates are applied.

## Negative/Neutral Results

- `trans12` fine-tune improved short-val E0 joint, but full-val best was only `72.9mm / 0.393deg / 20.9%`; it is not the main route.
- `gru_iters=3` was consistently worse than `gru_iters=2` in short sweeps.
- `aftertrans025` was less stable than `aftertrans0` for joint accuracy at outer4/5.

## Remaining Gap

- User target / STDLoc OldHospital reference: `119mm / 0.21deg`.
- Translation is already below the reference number in local refine, but rotation is still not competitive.
- The next bottleneck is not basic WLS or update scaling. It is likely query-map local correspondence and/or the learned hybrid rotation residual.

## Recommended Next Step

1. Add per-sample eval export for each outer iteration, including rotation/trans errors, flow EPE, confidence stats, and update norms.
2. Analyze which samples improve from outer3 to outer5 and which regress. This is needed before adding adaptive stopping or confidence-gated rotation updates.
3. Train/evaluate a rotation-focused local matcher or rotation residual head using outer5 as the validation protocol, not outer1.
4. Keep `aftertrans0 + outer5 + gru2` as the current inference baseline for all new comparisons.
