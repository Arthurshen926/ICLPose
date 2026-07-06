# RADIO-MATCHA Dense-Depth Measurement Fusion

## Branch Roles

- `vfm`: historical RADIO/VFM infrastructure and older localization baselines. Do not add new measurement-first code here.
- `codex/2dgs-radio-matcha-mainline`: active 2DGS Synthetic RADIO-MATCHA render-query mainline.
- `measurement_v1`: auxiliary high-resolution local measurement diagnostics and refinement code. It is not a replacement mainline.
- `codex/radio-matcha-dense-depth-fusion`: opt-in implementation branch that fuses RADIO/MATCHA coarse matches with local query-side measurement refinement and dense render-depth PnP.

## Active Mainline

The main research question remains:

```text
Can RADIO/VFM tokens, after feature selection/mapping and MATCHA-style training,
support render-query localization verification?
```

The pose pipeline should be interpreted as:

```text
RADIO/VFM feature selection and mapping
-> MATCHA-style render-query coarse correspondence
-> optional high-resolution query-side local measurement refinement
-> render pixel + dense render depth 3D correspondence
-> PnP / pose refinement
-> held-out evaluation
```

The RGB/FPN patch measurement branch is only an auxiliary fine-precision module.
It must not be used as evidence that RADIO tokens themselves form a metric
localization subspace.

## Dense-Depth Fusion Interface

The row-level fusion adapter is implemented in:

- `feature_extract/vfm/dense_depth_measurement_fusion.py`
- `feature_extract/tools/vfm/eval_dense_depth_measurement_fusion.py`

It consumes a RADIO/MATCHA `match_table.csv` and writes:

- `match_table.csv`
- `match_table.jsonl`
- `pose_rows.csv`
- `matrix_summary.tsv`
- `ablation_summary.tsv`
- `summary.json`

The normalized match table keeps these fields:

- `render_x/render_y/render_depth`
- `candidate_id/render_pose_id` when present
- `world_x/world_y/world_z`
- `world_xyz_source`
- `world_xyz_backproject_delta_m`
- `world_xyz_consistency_ok`
- `query_center_x/query_center_y`
- `query_refined_x/query_refined_y`
- `measurement_dx/measurement_dy`
- `measurement_cov_xx/measurement_cov_xy/measurement_cov_yy`
- `measurement_valid_prob`
- `radio_match_score`
- `local_cost_entropy`
- `measurement_search_radius_px`
- `query_gt_x/query_gt_y` when available for diagnostics
- `depth_valid`
- `pnp_inlier`

The geometry source is now explicit:

- `--geometry_source force_backproject`: recompute 3D from `render_x/render_y/render_depth` and the supplied render pose. This is the default CLI mode and is safest for dense-depth protocol checks.
- `--geometry_source prefer_world_xyz`: use materialized `world_x/world_y/world_z` when present. This is useful for legacy tables or full-query tables that already carry per-row 3D.
- `--geometry_source assert_consistent`: require both materialized world coordinates and render-depth backprojection to agree within `--world_xyz_consistency_threshold_m`.

Full-query or topK evaluation groups rows by `(query_id, candidate_id/render_pose_id)`
when candidate fields exist. A single `--render_pose_w2c_json` is allowed only for
one query/candidate group; otherwise the CLI fails loudly rather than applying one
render pose to unrelated rows.

Pose variants now support:

- `center`: use `query_center_x/query_center_y`
- `measurement` or `measurement_mean`: use explicit `query_refined_x/query_refined_y` or `measurement_dx/measurement_dy`
- `measurement_mode`: use `measurement_mode_dx/measurement_mode_dy` or `measurement_peak_dx/measurement_peak_dy`
- `oracle_if_gt_in_window`: use GT query pixel only when it is inside the local search window
- `oracle` or `oracle_all`: use all available `query_gt_x/query_gt_y`

The default evaluation is strict: `measurement` no longer falls back to `query_x/y`.
Use `--no_strict_measurement_schema` only for legacy debugging.

## Current Acceptance Gates

This branch currently closes the engineering interface, not the final accuracy
claim. The next full run must report:

- depth valid rate
- center baseline EPE
- measurement refined EPE
- measurement improve ratio vs center
- PnP success rate
- center / measurement / oracle PnP ablation
- uniform / weighted / covariance / oracle uncertainty solver comparison

The centimeter-level claim remains invalid until held-out real evaluation reaches:

- GT render median translation `< 3cm`
- reference topK median translation `< 10cm`
- `10cm/5deg > 80%`
- no test-set tuning

## Immediate Usage

Generate a RADIO/MATCHA match table from the main evaluator with:

```bash
python -m feature_extract.tools.vfm.eval_render_rgb_feature_keypoint_pose \
  ... \
  --save_match_table \
  --match_table_stage candidate
```

Then run dense-depth fusion postprocess:

```bash
python -m feature_extract.tools.vfm.eval_dense_depth_measurement_fusion \
  --match_table_csv /path/to/match_table.csv \
  --output_dir /path/to/dense_depth_fusion \
  --camera_width 640 \
  --camera_height 480 \
  --camera_model_id 1 \
  --camera_params fx fy cx cy \
  --geometry_source prefer_world_xyz \
  --variants center measurement measurement_mode oracle_if_gt_in_window oracle_all
```

If `world_x/world_y/world_z` are absent and all rows share one render pose, pass
`--render_pose_w2c_json` and keep the default `--geometry_source force_backproject`.
For multi-query or topK tables, provide per-row world coordinates or a future
per-candidate render-pose manifest; do not use one global render pose.

RGB patch measurement application now defaults to `center` / no-op. The branch
still runs and writes diagnostics, but it does not move the query pixel unless an
explicit ablation chooses a learned head. A mean or mode decoder must be promoted
to the PnP input only after controlled diagnostics show that it does not damage
good center measurements and improves the relevant residual bins. It still
writes mean/mode/direct diagnostics:

- `measurement_mean_dx/measurement_mean_dy`
- `measurement_mode_dx/measurement_mode_dy`
- `measurement_direct_dx/measurement_direct_dy`
- `measurement_peak_dx/measurement_peak_dy`

Use `--prediction_head likelihood_mean`, `--prediction_head likelihood_mode`, or
`--prediction_head direct` only for explicit decoder ablations or after the
conservative no-op / validity gates show that an update is safe.

Training and diagnostics now report residual-bin metrics, baseline EPE per bin,
worsen ratio, and `support_patch_source_audit`. These are protocol guards, not
accuracy claims.

## Full-182 Postprocess Snapshot

Generated outputs:

- GT render:
  `output/vfm/stage_r_matcha_joint/oldhospital/radio_matcha_dense_depth_fusion_v1/full182_gt_continuous_seed2026062401/`
- GT render RANSAC-only:
  `output/vfm/stage_r_matcha_joint/oldhospital/radio_matcha_dense_depth_fusion_v1/full182_gt_continuous_seed2026062401_ransac/`
- 0.25m near-GT render:
  `output/vfm/stage_r_matcha_joint/oldhospital/radio_matcha_dense_depth_fusion_v1/full182_gt_offset025_conservative25_headonly_120/`
- 0.25m near-GT RANSAC-only:
  `output/vfm/stage_r_matcha_joint/oldhospital/radio_matcha_dense_depth_fusion_v1/full182_gt_offset025_conservative25_headonly_120_ransac/`

Key observations:

- The old full-182 match tables did not contain learned measurement refinement.
  `center` and `measurement` are therefore identical in this snapshot.
- The postprocess projected `query_gt_x/query_gt_y` from `world_x/world_y/world_z`
  and the Cambridge query poses for all rows: no missing query poses or missing
  world points in the two full-182 tables.
- GT render has `center_epe_median_px = 1.409`, `measurement_improve_ratio = 0`.
  This is a coarse-center baseline, not a trained local measurement result.
- GT render with center RANSAC reaches median translation `0.00723m`,
  P90 `0.01785m`, and `3cm/1deg = 93.96%`.
- GT render with oracle query pixels reaches numerical-zero pose error, confirming
  dense-depth backprojection and PnP geometry are not the limiting factor.
- 0.25m near-GT render remains poor after RANSAC: median translation `0.244m`,
  `10cm/5deg = 3.30%`.

Conclusion:

```text
GT-basin dense-depth geometry is viable.
Near-GT coarse correspondences still do not recover the nonzero residual flow.
The next optimization target is query-side measurement/correspondence residual,
not another learned pose scorer.
```
