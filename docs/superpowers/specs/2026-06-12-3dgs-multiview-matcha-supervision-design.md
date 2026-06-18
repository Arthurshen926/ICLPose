# 3DGS Multiview MATCHA Supervision Design

## Goal

Build clean multi-view supervision for the RADIO-adapted MATCHA training path using only 3DGS-rendered geometry and known camera poses.

## Non-Goals

- Do not use current matcher outputs as positive labels.
- Do not use RADIO descriptor similarity, MATCHA scores, or any model prediction to decide positive correspondences.
- Do not replace the RADIO-adapted MATCHA architecture with official MATCHA DIFT/DINOv2 features.

## Builder API

Add a geometry-only builder that takes:

- a query view: camera, pose, rendered depth, optional alpha
- a render view: camera, pose, rendered depth, optional alpha
- additional support views: camera, pose, rendered depth, optional alpha
- query/render feature-grid sizes
- MATCHA coarse-supervision thresholds

The builder seeds candidate points from render grid cells, backprojects those cells through render depth into world space, projects the same world points into the query view, validates query depth/alpha/roundtrip consistency, and then validates visibility in support views by projecting the same world point and checking support depth/alpha agreement.

## Output

The output is a `MatchaCoarseSupervision` with `source="geometry_3dgs_multiview"`. It contains query/render cell indices, image coordinates, 65-bin offset labels, soft offset labels, confidence targets, and uncertainty values. Confidence is geometry-derived from pair roundtrip quality and support-view agreement.

## Streaming Integration

The streaming trainer remains pairwise by default. New CLI options enable multiview supervision explicitly:

- `--multiview_supervision_support_views N`
- `--multiview_supervision_min_support_views N`
- `--multiview_supervision_depth_tolerance_m VALUE`

When enabled, the trainer selects deterministic nearby GT-pose support views, renders their 3DGS depth/alpha, and calls the multiview builder. If no support views are requested, the existing depth+pose builder is used unchanged.

## Tests

Tests cover:

- multiview builder returns `geometry_3dgs_multiview` supervision
- support views increase confidence and filter candidates with insufficient support
- depth-inconsistent support views are rejected
- streaming CLI exposes the multiview knobs while defaulting to pairwise behavior
