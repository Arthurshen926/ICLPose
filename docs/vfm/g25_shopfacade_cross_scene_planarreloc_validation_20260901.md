# ShopFacade cross-scene PlanarReloc-style validation

## Outcome

The plane route is viable on a second Cambridge scene.  A map built only from the 20
actual MAtCha `seq2` mapping cameras localizes all 103 `seq1`/`seq3` official-test
queries with a single finite-plane retrieval/PnP implementation.  MoGe3 supplies query
plane masks only; query depth and scale are never passed to PnP.

The final machine report is
`output/g25_pose_transport/planar_map_shopfacade_cross_scene_v1/shopfacade_cross_scene_planarreloc_validation_v2.json`
(file SHA256 `50d09b1fb725e08004eb82fd1f8ab2c805baf831d942d547ac362bec364a1677`,
content SHA256 `be29001208db9957a779a50ff792fb30cbcbeff8079a5ba3034caedc8e69ce05`).

| operating point | median t/R | 25cm/2deg | 50cm/5deg | 1m/10deg | 2m/45deg |
|---|---:|---:|---:|---:|---:|
| RADIO plane Top1 | 0.140 m / 0.754 deg | 74.76% | 85.44% | 95.15% | **99.03%** |
| RADIO plane Top5 | **0.111 m / 0.596 deg** | **82.52%** | **89.32%** | **97.09%** | 98.06% |

`seq3` is especially strong: Top5 gives 67/67 at 1m/10deg, 66/67 at
25cm/2deg, median 9.4 cm / 0.53 deg.  `seq1` is the harder extrapolation route:
Top1 gives 35/36 at 2m/45deg, while Top5 improves median and fine recall but introduces
one extra coarse failure.  Therefore Top1 is the current robust basin operating point;
Top5 is a fine-accuracy branch, not an unconditional replacement.

## Map representation

The raw 1,294,373-surfel MAtCha 2DGS is converted directly to a strong
`PrimitiveSurfaceTable`; no parent/child/voxel identity is consumed.  Rendering the map
from the exact 20 mapping cameras produces 598 local plane observations.  Cross-view
fusion with the pre-existing `minimum_views=2` rule yields:

- 90 finite plane instances;
- 56 planes at least 1 m2, 11 at least 5 m2, and 7 at least 10 m2;
- median finite support area 1.474 m2;
- median plane-fit RMS 2.65 cm;
- 65,538 assigned surfels (6.63% of the opacity-filtered global cloud).

The small assigned-surfel fraction is not the held-image coverage.  It is the fraction
of the whole, cluttered global surfel cloud that is both observed by the sparse 20-view
mapping inventory and accepted as a repeated finite plane.  Rendered mapping-view plane
coverage is 47.90%.  Query plane coverage is 54.53% on seq1 and 65.91% on seq3.

The attempted alternative, a global 16-NN connected-component extraction over all raw
surfels, is KILL as the primary map builder: bridge chains create a 368,476-surfel giant
component spanning distinct facade/ground/edge surfaces, making sequential plane fitting
both physically wrong and computationally pathological.  View-conditioned finite plane
observations plus cross-view fusion are the correct abstraction for this outdoor map.

## Fixed bugs and contracts

1. ShopFacade RADIO is native `1280 x 68 x 120`, not the St Mary's `1280 x 36 x 64`.
   The old code rejected it.  Visibility masks, query plane masks, 2D token centres and
   2DGS depth lifting now use an explicit dynamic token-grid contract.  Arbitrary masks
   are area-projected and quantized to the same normalized 0..16 support convention.
2. The 68x120 token grid is bound into the visibility atlas, plane field, observation
   bank, plane ranking, frozen correspondences and final report.  Shape/hash mismatch is
   fail-closed.  The exact 36x64 path remains bit-compatible.
3. Query calibration is now built in a separate process from RADIO plus fixed mapping
   calibration only.  It does not parse a pose/GT file or open any pose-bearing NPZ
   member.  Poses are opened only after correspondences and PnP poses are frozen.
4. `inlier_source_view_count` previously counted plane-observation rows rather than
   unique mapping camera names.  This diagnostic-only bug is fixed; poses and recalls
   were unchanged and final v3 reports were regenerated.

Targeted validation: 19 tests passed, all modified Python compiled, and
`git diff --check` passed.

## Visual evidence

- mapping plane observations, 12 uniformly selected views:
  `output/g25_pose_transport/planar_map_shopfacade_cross_scene_v1/shopfacade_rendered_plane_observation_grid_v1.png`;
- multi-view fused plane map:
  `output/g25_pose_transport/planar_map_shopfacade_cross_scene_v1/shopfacade_fused_planar_map_minviews2_summary_v1.png`;
- 8 uniformly selected seq3 queries with MoGe3 plane boundaries and frozen PnP matches:
  `output/g25_pose_transport/planar_map_shopfacade_cross_scene_v1/shopfacade_query_plane_pnp_uniform8_v2.png`.

## Remaining issues and next action

- Seq1 cameras are farther from the sparse mapping inventory (nearest mapping-camera
  median 2.96 m versus 1.73 m for seq3) and have lower query-plane pixel coverage.  This
  is now the primary map/inventory limitation, not voxel identity.
- Query plane extraction still fragments the facade into about 31.7 finite regions per
  image and is CPU-expensive.  It is adequate for matching but should be replaced by a
  faster region merger after the pose operating point is frozen.
- Top5 improves fine accuracy but can add outlier basins.  A branch selector must be
  calibrated on mapping/development data; it must not be tuned on these 103 test labels.
- The result is a strong bounded cross-scene validation, not production promotion.  A
  third scene and a source-view densification ablation are needed before changing the
  default backend.

## Frozen 60-view map-density ablation

The source-view ablation was subsequently run without changing any plane extraction,
fusion, RADIO matching, homography, or PnP threshold.  A rounded-linspace inventory of
60 `seq2` mapping cameras was sealed before reusing the 103 test queries.  Relative to
the 20-view map, it produced 1,722 rather than 598 rendered plane observations, 217
rather than 90 fused planes, 188,620 rather than 55,267 metric mapping tokens, and
assigned 11.26% rather than 6.63% of opacity-filtered global surfels.  Median fit RMS
remained unchanged at 2.64 cm, so the gain is coverage rather than looser geometry.

The label-free Top1 operating point improved to:

| inventory | median t/R | 25cm/2deg | 50cm/5deg | 1m/10deg | 2m/45deg |
|---|---:|---:|---:|---:|---:|
| 20 mapping views | 0.140 m / 0.754 deg | 74.76% | 85.44% | 95.15% | 99.03% |
| 60 mapping views | **0.088 m / 0.503 deg** | **86.41%** | **93.20%** | **99.03%** | **100%** |

On the harder `seq1`, Top1 improved from 0.370 m / 1.63 deg and 31/36 at
1m/10deg to 0.179 m / 0.85 deg and 35/36; all 36 pass 2m/45deg.  On `seq3`,
Top1 is 0.076 m / 0.44 deg and all 67 pass 1m/10deg.  This is direct causal
evidence that the prior ShopFacade residual was dominated by sparse mapping support.

Top5 is no longer the preferred branch on the dense map: it has nearly identical
median translation but lower fine and 1m/10deg recall because extra similar finite
planes add ambiguity.  The frozen default for this ablation is therefore Top1; no
test-label branch selector was fitted.  The machine report is
`output/g25_pose_transport/planar_map_shopfacade_dense60_v1/shopfacade_cross_scene_planarreloc_dense60_validation_v1.json`
(file SHA256 `01b69a6e9cdb889cd68ee5eae08844db8e872b8f2d0bc29209c00d67d0cb831a`,
content SHA256 `80d81cd2a6a064a90d3b25747ba3e2241cdd5a0d6bf00a02adc7cd7d0237f003`).

## Full 231-view saturation control

The same frozen pipeline was finally run with all 231 `seq2` mapping views.  This is a
map-density upper-bound control, not a new tuned operating point.  It yields 6,711
rendered observations, 612 fused planes and 808,047 metric mapping tokens.  Assigned
global surfels rise to 16.72%, but median finite-plane area falls to 0.167 m2 while fit
RMS remains 2.64 cm.  Thus the added views recover real support and also expose a clear
map-fragmentation problem.

| inventory / branch | median t/R | 25cm/2deg | 50cm/5deg | 1m/10deg | 2m/45deg |
|---|---:|---:|---:|---:|---:|
| 60-view Top1 | 0.088 m / 0.503 deg | 86.41% | 93.20% | **99.03%** | 100% |
| 231-view Top1 | 0.088 m / 0.545 deg | 88.35% | 91.26% | 97.09% | 100% |
| 231-view Top5 | **0.078 m / 0.350 deg** | **91.26%** | **97.09%** | **99.03%** | 100% |

Full-view Top1 is less robust on `seq1` because a single query region must choose among
many more similar small plane instances.  The predeclared Top5 branch recovers that
coverage and gives the highest fine accuracy, including 67/67 `seq3` queries within
25cm/2deg.  The practical conclusion is two operating points: 60-view Top1 is the
lighter, less ambiguous default; 231-view Top5 is the current accuracy upper bound.
The next map-side task is deterministic coplanar instance merging/deduplication on
mapping-only data, not additional query-depth use or more mapping cameras.

A source-only fragmentation audit supports that diagnosis.  At a conservative
3deg/5cm coplanarity test, the 20/60/231-view maps contain respectively 22/51/184
planes in multi-instance coplanar families.  The fraction of planes below 0.25 m2
grows from 10.0% to 30.4% to 56.5%, while median convex-boundary-area / exact-support-
area grows from 2.45 to 4.40 to 8.70.  Convex hulls therefore become increasingly poor
finite-support summaries at full density; exact primitive membership remains the
authority.  A production merger must use exact support adjacency and an aggregate
refit, not convex-hull overlap or an infinite-plane identity alone.

The full-view report is
`output/g25_pose_transport/planar_map_shopfacade_dense231_v1/shopfacade_cross_scene_planarreloc_all231_validation_v1.json`
(file SHA256 `2e2a9c702964ce8ed37450a74b964d6a918af04d3ebfb7b387df7fd653afb9e5`,
content SHA256 `214fc174373e49b38bba4af9eaf985a5438a648f884efc43ea60460ba6747862`).

## Strict coplanar merger correction

The first covisibility merger used single-linkage unions.  A mapping-only audit found
that chaining could produce a component with 28.85 degrees of internal normal spread,
despite every accepted edge being below 6 degrees.  That artifact is invalidated.
The corrected merger requires complete-linkage coplanarity between every pair of
original source planes before each union, preserves exact primitive membership, and
refits from accumulated rendered-depth moments.  Its ShopFacade 231-view map has 358
planes rather than 612 (not the invalid 290-plane result); maximum internal spread is
5.997 degrees and maximum reciprocal distance is 4.988 cm.

On ShopFacade the strict merger is useful but not a universal replacement.  At Top1 it
raises 50cm/5deg from 94/103 to 97/103 and 1m/10deg from 100/103 to 103/103, while
median translation is unchanged.  At Top5 it exchanges individual successes and is
slightly worse in median translation.  A same-view view-balanced canonical RADIO mean
was also tested and killed: it collapses viewpoint-specific appearance and degrades the
first held route to 0.237 m / 0.967 degrees.

## Unseen third scene: OldHospital

OldHospital is a fully route-disjoint third-scene test.  The 2DGS map and RADIO mapping
observations use train routes `seq1/2/3/5/6/7/9`; all 182 queries come from unseen test
routes `seq4/8`.  Query MoGe3 is used only to segment plane regions.  Query intrinsics
come from a pose-free camera inventory and test poses are opened only after RADIO
rankings, 2D-3D correspondences, and PnP poses are frozen.

The initial MAtCha inventory had only 20 mapping views.  It produced 68 planes, 235
retained plane observations, 34,409 metric mapping tokens, and only 8.01% assigned
primitive support.  Its aggregate Top1 result was 1.125 m / 1.936 degrees, with
54/182 at 50cm/5deg, 87/182 at 1m/10deg, and 107/182 at 2m/45deg.  Strict merging alone
was mixed and therefore did not solve the scene.

A source-only nested route-balanced plan retained those 20 views and expanded to 60
and 120 views, with 8--9 and 17--18 views per train route respectively.  No test route,
pose, or score was read when selecting it.  The map-side curve is:

| mapping views | raw planes | observations | assigned primitives | assigned fraction |
|---:|---:|---:|---:|---:|
| 20 | 68 | 235 | 55,614 | 8.01% |
| 60 | 255 | 1,640 | 101,087 | 14.56% |
| 120 | 472 | 3,291 | 130,436 | 18.79% |

The 120-view runtime bank contains 262,446 valid metric RADIO-token observations, 7.6
times the 20-view bank.  The strict complete-linkage merger reduces 472 planes to 222
without changing observation or primitive support.  The held results are:

| branch | median t/R | 10cm/1deg | 25cm/2deg | 50cm/5deg | 1m/10deg | 2m/45deg |
|---|---:|---:|---:|---:|---:|---:|
| 20 raw Top1 | 1.125 m / 1.936 deg | 6 | 30 | 54 | 87 | 107 |
| 120 raw Top1 | 0.532 m / 0.972 deg | 8 | 46 | 85 | 116 | 139 |
| 120 raw Top5 | 0.511 m / 0.935 deg | 10 | **51** | 89 | 129 | 149 |
| 120 merged Top1 | 0.541 m / 1.083 deg | **11** | 46 | 82 | 113 | 143 |
| 120 merged Top5 | **0.444 m / 0.853 deg** | 7 | 42 | **97** | **132** | **158** |
| 120 merged Top5 H3 control | 0.459 m / **0.837 deg** | 10 | **51** | 93 | 131 | 153 |

The causal conclusion is strong: sparse map observation coverage, not a hidden voxel or
LoFTR dependency, was the dominant OldHospital failure.  Among 179 queries usable in
both 20-view and 120-view raw Top1, translation improves for 134 and rotation for 141.
The merger and H3 controls remain mixed: merged Top5 is best at medium/coarse thresholds,
raw Top5 and H3 retain more strict hits.  They must remain explicit branches until a
label-free selector is frozen on separate development data.

Residual coarse failures have much lower pose-free support than successes: median 88.5
PnP inliers, 6 query regions, 17 source views, and 13.6% query-plane coverage, versus
382, 15, 32.5, and 39.2% for 2m/45deg successes.  A smaller set of high-inlier gross
failures remains, consistent with repeated-facade ambiguity.  PnP already uses RANSAC
and LM; the next evidence-backed tasks are more source-view coverage and better query
plane/support coverage, not another unconstrained pose optimizer.

The sealed comparison is
`output/g25_pose_transport/planar_map_oldhospital_cross_scene_v1/oldhospital_mapping_density_plane_pnp_comparison_v1.json`
(file SHA256 `ca261e666d61ad4a8c9bdd98378144d977dcee9d958a823abddf57489594d9bc`,
content SHA256 `550388cc6238e90b60d8321fb73c541de3c59968fb3fbfd5cfcc11731f7061aa`).
Map and query visualizations are under
`output/g25_pose_transport/planar_map_oldhospital_cross_scene_v1/visualizations_v1/`.

### Incremental 240-view saturation check

The nested source-only plan was extended to 240 views (34--35 per train route).  The
first 120 IDs are byte-identical to the prior plan; only the additional 120 contributor
and plane-observation files were rendered, then combined by exact hard links with a
per-member hash audit.  No test label affected the extension.

The map is still growing: observations increase from 3,291 to 6,666, assigned
primitives from 130,436 to 158,765, and assigned fraction from 18.79% to 22.87%.
This +4.08 percentage-point gain is nearly the same as 60→120 (+4.23 points).  Raw
plane count grows to 788 and median area falls to 0.370 m2; strict complete-linkage
reduces it to 365 while preserving all 4,835 retained runtime observations and 561,263
metric RADIO-token points.

| branch | median t/R | 10cm/1deg | 25cm/2deg | 50cm/5deg | 1m/10deg | 2m/45deg |
|---|---:|---:|---:|---:|---:|---:|
| 120 raw Top5 | 0.511 m / 0.935 deg | 10 | 51 | 89 | 129 | 149 |
| 120 merged Top5 | 0.444 m / 0.853 deg | 7 | 42 | 97 | 132 | 158 |
| 240 raw Top5 | **0.401 m / 0.746 deg** | **14** | 57 | **107** | **134** | **160** |
| 240 merged Top5 | 0.451 m / 0.780 deg | 12 | **60** | 100 | 132 | 157 |

Thus the current default accuracy branch is 240-view raw Top5.  The merger is not a
monotone localization improvement at high density and remains an auxiliary compact-map
branch.  The gain from 120→240 is real but smaller than 20→120.  The remaining 22
2m/45deg failures still have very low pose-free support: medians are 120 PnP inliers,
7 query regions, 19 source views, 16.3% inlier hull, and 12.6% query-plane coverage,
versus 474, 17, 50, 46.0%, and 39.1% for successes.  The next primary task should
therefore improve query plane/support coverage and repeated-facade disambiguation before
another blind doubling to 480 views.

The updated sealed report is
`output/g25_pose_transport/planar_map_oldhospital_cross_scene_v1/oldhospital_mapping_density_plane_pnp_comparison_v2.json`
(file SHA256 `0fcab48b992f2096ebf898ac43911b78556108d8b91fe70b5c24db99ffc53c75`,
content SHA256 `9e2e9c5d311707903e84e53aaec320b6d9408d1cde13fd3a422d7e63aace8a1c`).
