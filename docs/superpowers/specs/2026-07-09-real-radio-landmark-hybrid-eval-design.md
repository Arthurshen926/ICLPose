# Real RADIO Landmark Hybrid Eval Design

## Goal

Evaluate a two-stage localization path that uses real-image RADIO features only:

1. map query RADIO tokens with the trained joint feature mapper,
2. match mapped query descriptors directly to aggregated multi-view 3D landmarks,
3. optionally refine query 2D coordinates and confidence with the RGB patch measurement branch using real owner-view observations,
4. solve absolute pose with PnP and report the same pose metrics as the current real-support eval.

## Scope

This is an evaluation-only change. It does not retrain selector/coarse/measurement and does not use render/2DGS data.

The contrast is:

- current real-support path: query/reference cell proposals -> nearest support SfM observation -> PnP;
- landmark hybrid path: query token -> landmark descriptor/xyz -> optional owner RGB measurement -> PnP.

## Data

- Query features come from `TokenBankManifest` records.
- Landmark features come from a `SelectedTrackFeatureBank` built from real train-image RADIO tokens and SfM track observations.
- Owner-view measurement uses the same SfM track observations to choose a real reference image and scaled observation coordinate.
- Pose ground truth and cameras use the Cambridge/Colmap files already used by current pose eval.

## Method

Landmark matching reuses `match_query_tokens_to_landmarks` and `evaluate_query_poses`. Joint projection of landmark bank features is supported by projecting each raw landmark vector as a one-cell feature map through `MatchaStyleJointModel.forward_feature_map`, matching the query mapper behavior for the single-input residual adapter checkpoint.

Measurement is optional and must not change the 3D point. It only updates `QueryTo3DMatch.xy`, `pnp_soft_score`, `patch_offset_confidence`, and `measurement_sigma_px`.

## Outputs

The CLI writes:

- `pose_rows.jsonl/csv`
- `matches_2d3d.jsonl`
- `measurement_rows.jsonl` when measurement is enabled
- `summary.json` with query count, match count, PnP success, pose recall, and method metadata.

## Known Limitation

For `radio_dual_attention` checkpoints, projecting independent landmark vectors discards spatial attention context. The first controlled experiment should use the single-input residual adapter checkpoint from `real_radio_joint_referenced_v1`.
