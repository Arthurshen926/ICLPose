# Surface-centric RADIO plane localization upgrade (2026-09-02)

## Frozen method direction

The mainline is a source-image-free, retrievable mid-level map:

1. retrieve finite physical planes/charts with RADIO;
2. match query RADIO tokens to anonymous modes in a canonical metric plane-UV field;
3. generate several plane-consistent pose hypotheses;
4. select/verify hypotheses with query MoGe3 geometry;
5. refine pose with balanced surface reprojection, plane normals, plane offsets and one latent depth scale.

PnP remains a robust initializer and a control, not the claimed map representation.  Runtime artifacts explicitly store neither source RGB nor source-view identity.  The intended final objective is probabilistic surface reprojection/registration; points are temporary numerical samples of finite surfaces.

## Implemented corrections

### Plane-specific geometry lifting

`evaluate_goal_maplet_radio_plane_pnp.py` now supports continuous source-token-to-plane homography coordinates and exact camera-ray/plane intersection.  This removes the conceptual dependency on rendered token depth.  Directly replacing the current point lift with the large fused-plane surface, however, degraded seq10 from 0.237 m / 0.624 degrees to 0.440 m / 0.971 degrees.  This is a useful KILL: the current large planes are not yet locally accurate enough to be used as hard metric surfaces.

### Plane-balanced surface solver

The solver now caps quadratic evidence mass per plane, never amplifies sparse planes, and robustly refines a PnP seed by finite-surface reprojection.  It has a fail-closed support-retention gate.  Alone it improves median error but has mixed recall, so it is retained as a proposal branch rather than a replacement.

### MoGe3 is now used by pose, not only visualization

`refine_goal_maplet_plane_pose_with_moge3.py` jointly optimizes pose and one latent MoGe3 scale from:

- balanced surface reprojection;
- matched plane normal residuals;
- matched plane offset residuals;
- a weak metric-scale prior.

Associations and the acceptance decision are frozen before query pose labels are opened.  Metric depth is not assumed to be ground truth; scale is optimized and guarded by label-free residual monotonicity.

On seq10, Top10 RADIO PnP -> MoGe3 plane/scale refinement changed median 0.2367 m / 0.6237 degrees to 0.2200 m / 0.5066 degrees, improved 0.25 m / 2 degrees from 52.27% to 54.55%, and did not reduce 2 m / 45 degrees or 1 m / 10 degrees recall.  Plane-balanced initialization followed by the same fixed MoGe3 refinement reached 0.2187 m / 0.5011 degrees, 95.45% at 2 m / 45 degrees and 89.77% at 1 m / 10 degrees.

On an interleaved 88-query seq13 held shard, the unchanged MoGe3 plane/scale optimizer improved:

- median: 0.2187 m / 0.6658 degrees -> 0.1834 m / 0.5860 degrees;
- 2 m / 45 degrees: 82.95% -> 84.09%;
- 1 m / 10 degrees: 80.68% -> 81.82%;
- 0.25 m / 2 degrees: 57.95% -> 63.64%;
- 0.10 m / 1 degree: 25.00% -> 27.27%.

This is the strongest evidence so far that MoGe3 geometry and latent scale add real pose information rather than merely correlate with pose quality.

### MoGe3 candidate selection

`select_goal_maplet_direct_plane_pnp_by_moge3_normal.py` selects frozen Top5/Top10 candidates by rendered-map/query-MoGe3 normal agreement at 20 degrees.  It consumes no pose labels and stores no source RGB/view identity.

- seq10: 0.2087 m / 0.5178 degrees, 96.59% at 2 m / 45 degrees, 90.91% at 1 m / 10 degrees, and 63.64% at 0.25 m / 2 degrees;
- seq13 77-query divergent complement: 1.5345 m / 5.5937 degrees, 53.25% at 2 m / 45 degrees and 15.58% at 0.25 m / 2 degrees, improving the previous inlier-ratio choice and matching the frozen candidate-pool 2 m oracle.

Raw metric depth scale was not a reliable selector.  Normal consistency generalizes; scale-free/affine depth remains useful as a diagnostic and potential residual, not a standalone branch score.

### Source-view-free canonical RADIO field

The previous multi-prototype atlas had a concrete bug: diverse RADIO modes in one texel all inherited the same mean UV, creating feature/geometry pairs that were never jointly observed.  V3 preserves each anonymous mode's observed metric UV.  On seq10 this improved the canonical four-mode map from roughly 0.626 m / 1.321 degrees to 0.382 m / 0.898 degrees and 2 m recall from 78.41% to 87.50%.  It also beats a single-prototype canonical map in median and coarse recall.

V4 additionally stores, per anonymous prototype:

- metric plane UV;
- RADIO descriptor;
- surface-to-camera unit direction;
- metric observation range;
- support counts and anonymous feature-mode rank.

It still stores no RGB, source path, or source-view identity.  Its descriptor/UV/base topology is bit-identical to V3; the new nuisance coordinates enable pose-conditioned matching without reverting to image place recognition.  Correspondence V2 explicitly binds the prototype row and carries the query plane visible fraction for future soft core/halo weighting.

With a broad, frozen same-hemisphere and 1/3x--3x range gate, followed by a support-preserving second PnP, the canonical field improves on seq10 from 0.3824 m / 0.8979 degrees to 0.3373 m / 0.7834 degrees.  Recall changes are 87.50% -> 89.77% at 2 m / 45 degrees, 78.41% -> 86.36% at 1 m / 10 degrees, 59.09% -> 68.18% at 0.5 m / 5 degrees, and 31.82% -> 37.50% at 0.25 m / 2 degrees.  This is a substantial gain from a genuinely source-view-free mid-level map; it does not retrieve or reopen a mapping image.

Applying the same MoGe3 plane/scale optimizer after this canonical-map pose improves it again to 0.2686 m / 0.6769 degrees, with 90.91% at 2 m / 45 degrees, 87.50% at 1 m / 10 degrees, 73.86% at 0.5 m / 5 degrees, and 46.59% at 0.25 m / 2 degrees.  Thus view/range conditioning and query geometry are complementary.  The resulting pipeline is already much closer to the view-centric control while satisfying the intended map-storage story.

The unchanged chain was then replayed on the 88-query interleaved seq13 held shard.  Canonical raw -> view/range-conditioned -> MoGe3 plane/scale changed:

- median: 0.4241 m / 1.2784 degrees -> 0.3152 m / 0.9756 degrees -> 0.2524 m / 0.7419 degrees;
- translation P90: 9.056 m -> 1.850 m -> 1.372 m;
- 2 m / 45 degrees: 84.09% -> 89.77% -> 89.77%;
- 1 m / 10 degrees: 76.14% -> 86.36% -> 87.50%;
- 0.5 m / 5 degrees: 55.68% -> 68.18% -> 79.55%;
- 0.25 m / 2 degrees: 35.23% -> 38.64% -> 47.73%.

This held result is especially important: anonymous view geometry removes catastrophic canonical-map outliers, while MoGe3 recovers fine metric pose accuracy.  The final source-image-free branch is now close to the view-centric control and exceeds it at coarse recall, although it still trails at the finest thresholds.

Hard core-only homography fitting was tested and KILLed because it removed too much support.  The next form is soft: core has high information weight, verified halo has lower weight, and boundary tokens cannot dominate a plane.

A first parameter-free square-root visible-fraction weighting was also tested in the final MoGe3 solve.  It slightly improved median translation and 0.1 m recall but reduced 0.25 m and 0.5 m recall, so it remains an explicit ablation and uniform weighting remains the default.  The evidence should instead enter plane sampling/uncertainty, not blindly rescale every final residual.

## Current claim boundary

The method is not yet at its upper bound.  The main remaining gap is between the strong view-centric control (about 0.237 m) and the source-view-free canonical field (about 0.382 m before MoGe refinement).  This is now an appearance/conditioning problem, not a reason to return to source images.

The following are not promotion claims:

- the current St Mary's map and seq10 route are development/historical controls;
- PnP is still the initializer;
- the large fused planes are not accurate enough for hard continuous intersection;
- query MoGe3 scale is a latent variable, not ground-truth depth;
- current canonical prototypes still need view/range-conditioned scoring and better local surface geometry.

## Next frozen priorities

1. Evaluate V4 pose-conditioned anonymous mode gating with broad hemisphere/range guards, retaining the V3 result exactly as the control.
2. Replace hard core/halo selection with visible-fraction weights in plane-balanced sampling and robust surface residuals.
3. Upgrade each large plane from a single infinite equation to a finite chart with local residual height/uncertainty; this should make continuous plane lifting viable.
4. Add a direct plane homography bundle branch and a probabilistic query-token-to-chart-UV likelihood branch.  Compare both against PnP initialization under an identical retrieved plane inventory.
5. Use MoGe3 relative depth ordering and affine log-depth residuals across multiple matched planes, with scale optimized jointly and a scale observability/rank gate.
6. Validate the combined canonical-field + MoGe3 optimizer on complete held routes and on OldHospital/ShopFacade, without retuning constants.

The paper-level story is therefore: **finite chart retrieval -> anonymous RADIO surface field -> probabilistic chart-UV correspondences -> plane-balanced surface pose -> MoGe3 normal/depth/scale refinement**.  It is explicitly neither image retrieval followed by local matching nor a conventional point-map localization pipeline.

## 2026-09-03 local-surface-coordinate follow-up

The anonymous RADIO/UV atlas now optionally retains the signed height of each
view-mode above its ideal finite plane, plus within-view texel uncertainty. It
still stores neither RGB nor source-view identity. Across 95,449 prototypes the
absolute height median/P90/P99 is 0.023/0.065/0.123 m; only 0.19% exceed 0.20 m.
A robust companion applies height only inside the planar map's already frozen
0.10 m fusion tolerance and otherwise falls back to the ideal plane.

On seq10, the untrimmed local surface followed by anonymous-view and MoGe3 plane
refinement reaches median 0.268 m / 0.610 degrees and recalls
94.3/88.6/75.0/47.7% at 2 m/1 m/0.5 m/0.25 m. A symmetric, label-free selector
grades untrimmed and robust poses on both geometry inventories and changes 11/88
queries. It retains 2 m and 0.5 m recall, raises 1 m recall to 89.8%, and gives
0.268 m / 0.588 degrees median error.

The selector was frozen and applied once to previously unused seq13 shard1.
Relative to the better single local-surface branch it raises 2 m recall
87.5% -> 88.6% and 1 m recall 83.0% -> 84.1%, while 0.5 m falls
70.5% -> 69.3%. It is therefore a useful coarse-robustness branch, not a
uniformly superior final estimator. It remains below the old view-centric
control at fine thresholds (81.8%/62.5% at 0.5 m/0.25 m), so source-free fine
localization remains open.

Direct MoGe3 token-block 3D to atlas Sim(3) registration was also implemented
and tested. Although it reduces its own robust 3D residual for accepted cases,
seq10 localization degrades to 0.466 m / 0.988 degrees and 47/88 at 0.5 m.
This route is KILL in its present form: RADIO correspondences and monocular
token depth do not yet define clean material-level 3D pairs. MoGe3 remains
useful through plane normals, offsets, and one latent scale, but not as
unrestricted per-token depth alignment.

## 2026-09-03 multi-scale chart-consistency follow-up

The original metric homography gate accepted a 1.0 m UV residual although the
atlas cell is only 0.5 m. Seq10 ablations at frozen 1.0/0.5/0.25 m gates show
that 0.25 m materially improves fine localization but loses some coarse cases.
The already label-free cross-inventory selector was therefore applied to the
1.0 m support-preserving branch and 0.25 m precision branch. It evaluates both
poses on both frozen correspondence inventories and selects the larger mean
unique-token inlier ratio, without query pose or GT.

On seq10 the selector splits exactly 44/44 and improves the single 1.0 m branch
from 0.268 m / 0.610 degrees to 0.237 m / 0.534 degrees. Recall becomes
96.6/90.9/78.4/52.3/11.4% at 2 m/1 m/0.5 m/0.25 m/0.1 m; every principal
threshold improves over the 1.0 m input branch.

The rule was frozen and applied once to the previously unopened seq13 shard2
(87 queries). It chooses 48 loose and 39 strict poses. Relative to the loose
branch, median improves 0.240 m -> 0.226 m, 2 m recall 87.4% -> 88.5%, 1 m
and 0.5 m remain 85.1% and 77.0%, 0.25 m improves 50.6% -> 54.0%, and 0.1 m
improves 10.3% -> 11.5%. The strict branch alone has better finest recall but
catastrophic P90 outliers; consensus recovers coarse robustness. This is a
held-directional GO for a two-scale plane-chart matching head.
