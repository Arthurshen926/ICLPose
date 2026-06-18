# RADIO MATCHA Fine Refactor Design

## Goal

Fix the remaining fine matching gap in the RADIO-adapted MATCHA path without weakening the already-correct geometry supervision boundary. Training must continue to use depth, pose, 3DGS alpha/depth, roundtrip checks, and multiview support. Matcher outputs must not become supervision.

## Current Failure

The current strict `cell_center` fine supervision uses render cell centers as geometry seeds. That keeps geometry clean, but it collapses render-side fine labels to one center bin. Query labels remain high entropy because projected query coordinates are continuous. The result is a misleading render fine accuracy of 1.0 and a weak query pair fine head.

The pair-MLP fine heads are not a faithful MATCHA fine mechanism in this setup because they only see two matched cell descriptors. MATCHA-style fine matching needs local evidence around the coarse match. In this repo, the logically consistent pieces are the existing local patch correlation loss and local correlation/search refiners.

## Design

### Fine Supervision Sampling

Add a geometry-only render sub-cell seed mode for streaming training. For each render feature cell, sample one deterministic sub-cell point inside the cell using the training record seed. The point is then passed through the existing 3DGS multiview supervision builder, so all depth, alpha, support-view, and roundtrip validation remains unchanged.

Keep existing `cell_center` and `render_alike` modes available for ablation. New RADIO-MATCHA training runs should use the geometry sub-cell mode.

Training summaries must report query/render offset-label entropy and center-bin fractions so label collapse is visible before full training.

### Fine Training Objective

Disable pair-MLP fine losses by default in streaming training. Keep the modules loadable for old checkpoints, but do not treat them as the primary fine objective.

Make local patch correlation the primary fine objective. It is geometry-supervised, symmetric, and local-window based. It teaches query-to-render and render-to-query descriptor maps to put the true corresponding cell at the local window center.

### Inference Fine Strategy

Do not rely on query-side pair-MLP refinement. Use local correlation/search refinement for render-side measurement, because PnP backprojects render depth to 3D. Query-side coordinates still matter as the 2D measurement, but the first retained inference improvement should be one that demonstrably improves pose metrics.

Add or use an explicit evaluation preset for RADIO-MATCHA local search: learned/blended confidence, coverage filtering, optional render-side local offset expansion, and local correlation refinement. It is a pose-backend strategy, not supervision.

### Evaluation Protocol

Every retained change must be evaluated in three tiers:

1. `gt`: tests matcher/fine quality when render pose is correct.
2. `gt_offset025`: tests local convergence from a small pose perturbation.
3. `reference_top5`: tests full localization from retrieval candidates.

Centimeter-level `gt` results cannot be used as evidence that full localization is solved.

## Success Criteria

- A training sample using geometry sub-cell seeds has non-collapsed query and render fine-label entropy.
- Streaming smoke training records `supervision_source=geometry_3dgs_multiview` and reports healthy entropy.
- Pair-MLP fine losses are off by default in the new RADIO-MATCHA recipe.
- Patch correlation remains active and reports meaningful accuracy.
- A full or representative evaluation improves at least one relevant pose tier without degrading the complete `reference_top5` tier enough to invalidate the change.

## Non-Goals

- Do not reintroduce matcher-derived supervision.
- Do not remove the clean depth/pose/3DGS multiview builder.
- Do not claim `gt` render results as full localization accuracy.
