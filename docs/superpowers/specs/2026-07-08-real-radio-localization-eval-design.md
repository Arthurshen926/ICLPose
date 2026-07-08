# Real RADIO Localization Evaluation Design

## Goal

Add a complete real-image evaluation protocol for the unified selector + coarse match + measurement localization module. The primary metric is absolute camera localization accuracy from PnP, reported per query image as translation and rotation error. The evaluation must not use rendered images or rendered-depth supervision; it uses real RGB images, real RADIO features, COLMAP sparse tracks, and Cambridge ground-truth poses.

## Current Gap

The existing real closed-loop evaluation writes coarse proposals and measurement refinements, then reports pair-level 2D errors when a proposal lands near a known CSV ground-truth reference point. That is useful for debugging but not a complete localization metric because it does not:

- associate closed-loop proposals to 3D landmarks,
- merge proposals from multiple support/reference images for the same query,
- solve PnP,
- compare the estimated pose with query ground truth,
- separate conditional coarse rank from global proposal recall.

## Evaluation Layers

The final eval should report three complementary layers.

1. Coarse conditional rank:
   For rows with known query/reference GT coordinates, measure the rank of the GT reference cell for the GT query cell. This evaluates selector + coarse descriptors without global proposal truncation. Metrics: top1/top5/top10/top32/top100, median rank, mean reciprocal rank, and mean GT score.

2. Closed-loop proposal and measurement diagnostics:
   Run `JointFeatureMapper + MatchaTopKCoarseMatcher + RGBPatchMeasurementAdapter` on real query/support image pairs. Report global proposal recall against available GT rows, coarse query EPE, measured query EPE, measurement improvement ratio, measurement confidence, and uncertainty.

3. Full pose localization:
   Convert measured proposals into `QueryTo3DMatch` records, solve PnP per query, and report absolute pose accuracy. This is the headline localization result.

## Full Pose Data Flow

For each query image:

1. Load all candidate real support pairs from the evaluation CSV.
2. Load query/support RGB and raw RADIO features lazily.
3. Map features with `JointFeatureMapper`.
4. Generate coarse real-image proposals with `MatchaTopKCoarseMatcher`.
5. Refine query/reference coordinates with `RGBPatchMeasurementAdapter`.
6. For each refined support/reference coordinate, find the nearest COLMAP observation in that support image within a configurable pixel radius.
7. Attach the matched COLMAP `point3D_id`, xyz, reprojection error, and track length.
8. Build `QueryTo3DMatch` using measured query xy as the 2D point and the support-associated xyz as the 3D point.
9. Deduplicate repeated query/track matches by best confidence.
10. Optionally apply spatial diversity selection before PnP.
11. Run `estimate_pose_pnp_ransac`.
12. Compare against Cambridge GT pose with `pnp_pose_error`.

The CSV row `track_id/support_track_id` is not trusted as the only 3D association because those IDs can differ. The pose eval uses nearest COLMAP observation from the actual support image coordinate, which matches the closed-loop proposal semantics.

## Metrics

Pose summary:

- query count, solved count, success rate,
- median and p90 translation error in meters,
- median and p90 rotation error in degrees,
- recall at 0.25m/2deg, 0.5m/5deg, and 5m/10deg,
- median match count, inlier count, and inlier ratio,
- PnP reprojection residual mean/median/p90,
- spatial coverage statistics for all matches and PnP inliers.

Bridge diagnostics:

- number of proposals, measured proposals, and 2D-3D associated proposals,
- nearest COLMAP association success rate,
- nearest support-observation distance median/p90,
- missing camera, missing pose, insufficient match, and PnP failure counts.

Artifacts:

- `proposals.csv/jsonl`: raw closed-loop proposals and measurements,
- `matches_2d3d.csv/jsonl`: proposal-to-track associations,
- `pose_rows.csv/jsonl`: one row per query with PnP result,
- `summary.json`: aggregate metrics and all runtime parameters.

## Components

Add reusable evaluation helpers under `feature_extract/vfm/localization/`:

- `pose_eval.py`: COLMAP observation index, nearest support observation lookup, proposal-to-2D3D conversion, PnP query aggregation, and summary metrics.

Add CLI:

- `feature_extract/tools/vfm/eval_real_radio_pose_localization.py`

The CLI should share the same model-loading path as `eval_real_radio_closed_loop.py` and accept explicit paths for:

- pairs CSV,
- image root,
- feature root,
- COLMAP sparse model dir,
- Cambridge query pose file,
- matcha joint checkpoint,
- measurement checkpoint,
- output dir.

## Error Handling

The eval should fail fast for missing required files or unsupported camera models. Per-query failures such as no 2D-3D associations, too few matches, missing pose, or PnP failure should be recorded in `pose_rows` and counted in `summary.json` instead of stopping the full run.

Image sizes and camera intrinsics must come from real image/COLMAP metadata, not hard-coded defaults. If the real image size differs from the COLMAP camera size, intrinsics are scaled with the existing `scaled_colmap_camera` logic.

## Testing

Add focused tests that do not require the full Cambridge dataset:

- nearest COLMAP observation lookup chooses the closest support observation and respects radius,
- proposal-to-2D3D conversion uses measured query/reference xy when available and falls back to coarse xy,
- duplicate query/track matches keep the best confidence,
- pose summary treats failed queries as failures in recall denominators,
- CLI argument parsing exposes the full pose-eval inputs and defaults.

Optional integration on mounted Cambridge data can be run manually after unit tests.

## Non-Goals

- No rendered RGB, rendered feature maps, or rendered depth.
- No retraining changes inside this eval step.
- No replacement of the current training loss; the eval should expose whether the current joint training is useful at pose level.
- No reliance on CSV `track_id` as the only ground-truth 3D association.
