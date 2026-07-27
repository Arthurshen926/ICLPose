# V6: 2DGS Maplet-Atlas Localization

V6 is the active research line. It localizes a query against a feature-bearing
2DGS map; it does not retrieve or match a stored reference image.

## Frozen method definition

```text
Stage A  query RADIO-final regions
         -> full-map maplet identity posterior + identity null

Stage B  region identities + canonical maplet geometry
         -> several region-level coarse SE(3) modes

Stage C  render selected canonical feature atlases at every coarse mode
         -> multi-scale local displacement distributions
         -> robust joint SE(3) updates, retaining multiple flow modes

Stage D  maplet-disjoint evidence
         -> accept, reject, or retain several pose modes
```

RADIO regions are area/context observations. Their centers are not keypoints
and must not be interpreted as exact 2D measurements of surface cells.
Consequently, regional-center PnP is a diagnostic decomposition only. It is
not the coarse estimator and never becomes the final estimator.

The target runtime map contains 2DGS geometry and bounded anonymous feature
sufficient statistics. It contains no mapping RGB, mapping image paths or IDs,
per-view descriptor list, SfM point/track, stable-anchor identity, ALIKE
descriptor map, RADIO-intermediate feature, or pairwise image matcher input.
A maplet may retain a small anonymous appearance mixture; this is not a bundle
of reference-image descriptors.

## P0 probability and representation contract

The identity and spatial posteriors are different random variables.

```text
p(identity m | query region)

p(surface mode k | identity m, query region, spatial atlas available)
```

The identity bank is always the full bank. A maplet absent from the spatial
atlas remains an identity candidate with:

```text
spatial_available = false
p(spatial null | m, q) = 1
```

Every available identity candidate has its own conditional spatial null:

```text
sum_k p(surface mode k | m,q) + p(spatial null | m,q) = 1
```

Mass removed by spatial NMS or mode truncation is transferred to that spatial
null. Retained modes are not renormalized. A representative geometry value is
a real MAP cell; a posterior mean of distinct 3D cells is forbidden.
Scene-level Top-K removal is transferred to identity null.

Identity-null and conditional-spatial-null probabilities come from a frozen,
trajectory-disjoint artifact fitted with unweighted Bernoulli NLL. Class
balanced classifier scores are not probabilities and are rejected by the
artifact loader. Conditional-spatial labels contain only true identities with
an available atlas; wrong identities are not relabelled as spatial nulls.

The compact identity map now retains canonical tangent frames as geometry.
This is required for an oriented projected-area model. Inferring a chart frame
from only a normal is explicitly non-canonical and is rejected by footprint
localization. Pose-vote artifacts hash both descriptor components and retrieval
geometry.

## Stage B implementations and status

Three coarse methods are present:

1. `anonymous_view_mode`: an image-free maplet/component pose mixture,
   conditioned on query-region position. The query/source bearing difference
   rotates each vote before symmetric SE(3) clustering. Repeated overlapping
   RADIO regions use max aggregation.
2. `maplet_footprint`: projected canonical maplet quads are dilated by the
   RADIO support footprint. Candidate identities are marginalized against an
   explicit unit-likelihood unmatched event, and CEM refines coarse modes.
3. `regional_center_pseudo_pnp`: a named diagnostic baseline. It treats a
   region center as a pseudo point, uses deterministic OpenCV RNG, and cannot
   be reported as production localization.

The footprint foreground term is an image-background density ratio. The older
unnormalized expression was bounded by one, so a match could never beat the
unmatched branch; that implementation was invalid. The older spherical
maplet approximation was also invalid because canonical chart axes and
extents were discarded. The current model projects an oriented canonical
quad and uses its dilated area.

Neither region-level method is production-qualified yet. In particular,
stored anonymous view modes are not a substitute for cross-trajectory pose
extrapolation.

## Stage C/D contract and current blocker

Canonical atlases use fixed clean-2DGS geometry and raster contributor IDs.
Mapping observations update only anonymous feature sufficient statistics,
support and uncertainty; they never move canonical XYZ. Rendering uses
perspective-correct area interpolation and selected-maplet z-buffering. The
analytic solver uses the radial-camera Jacobian, robust covariance weighting,
per-maplet balancing, and maplet-disjoint verification.

The current correlation null remains:

```text
fixed pair-null logit + atlas-uncertainty term
```

and still uses a random-unit-vector partition approximation. It is not the
required typed calibration over correlation peak, top-2 margin, entropy,
periodicity, spread, identity probability, visibility, occlusion and
matchability. Therefore Stage C/D is not runtime-qualified. The missing typed
outcomes are at least:

```text
wrong identity
spatial unavailable
out of search window
occluded
query unmatchable
valid flow / zero flow
```

The runtime path also does not yet close two or three render/correlate/update/
verify iterations. A one-step oracle experiment is a component diagnostic,
not end-to-end localization.

## StMaryChurch artifacts used by the corrected audit

The corrected map lineage uses the clean 2DGS geometry and these generated
artifacts:

- full identity map: `identity_maplets_full864_mapper_k4_frames_v2.npz`
  (864 maplets, 3,226 anonymous appearance components);
- separate spatial map: 783/864 identities have at least one spatial atlas
  component;
- pose statistics:
  `anonymous_pose_votes_full_identity_region_conditioned_k4_frames_v5.npz`;
- probability calibration:
  `probability_calibration_seq11_full_identity_frames_conditional_proper_v6.json`.

The frozen audit lineage is:

| Artifact | SHA-256 |
| --- | --- |
| `/root/StMaryChurch2dgs_clean.ply` | `4127e187090baa3ab1f21cdba7e3b0de8b51c0e177d29a425120964f18c506af` |
| canonical atlas geometry | `43cbc9c0effb2932baa449ab72b45e185ef6241b989b54321f3de13b39b92c48` |
| full identity map | `be3b0617a9ffc0521f544e631fced3a13ae9ae4c73a3eee41f6b24bd9030eceb` |
| separate spatial map | `fbe0f54ce14b1442097b971df963daf6cefcc4db133c642932489d613848811f` |
| anonymous pose statistics | `e112f3f94bc411052a2be118d61f88befafdee5ef13a94d6c7940d4d8ead5780` |
| probability calibration | `319626aef6b65997e61a5e50c8148d84ded7c09beccd55cb5af25d3f188bcd6c` |
| corrected strict-12 dashboard (`dashboard_bearing_corrected_view_vote_strict12_v13.json`) | `8d53f233e65f548f1b091cedc9a844969383ae9e5d218696684cb95c164be177` |

The calibration trajectory is `seq11`; the fixed strict audit trajectories are
`seq3`, `seq5`, and `seq13`. They are disjoint. The calibration artifact has:

| Posterior | Examples | Null prevalence | AUPRC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity null | 768 | 42.19% | 61.11% | 0.2283 | 0.0667 |
| spatial null, conditional on true identity + atlas | 608 | 12.34% | 18.50% | 0.1072 | 0.0130 |

The conditional spatial probability is calibrated in prevalence but has weak
ranking power. The near-perfect AUPRC obtained by mixing wrong identities into
spatial-null labels was a metric-definition bug and is invalid.

Pose-vote artifacts now record every mapping trajectory. By default the
evaluator rejects any query trajectory that overlaps either probability
calibration or pose-vote map construction. The
`--allow_reference_trajectory_diagnostic` escape hatch is diagnostic-only:
such a run is reference replay, not independent localization evidence. In
particular, the earlier `seq11` result produced from an all-training pose bank
cannot support a generalization claim.

The 128-view bake covers 787 of 848 geometry-bearing maplets, but only 47.38%
of valid canonical texels. Median coverage among baked maplets is 51.97%;
physical texel size has a 4.81 cm P90. Adaptive 16/32/64 charts, automatic
split of disconnected/non-planar maplets, and full-scene runtime occlusion
remain open.

## Corrected fixed strict audit

The promotion audit contains 12 deterministic queries sampled from the
official test trajectories. It is not the 530-query result.

After full identity/spatial decoupling, proper null calibration, region
conditioning, symmetric pose kernels, canonical-frame geometry and bearing
correction:

| Metric | Result |
| --- | ---: |
| identity probability conservation max error | about `1.2e-7` |
| conditional spatial conservation max error | about `6.9e-7` |
| dominant identity recall @1 / @5 / @64 | 29.92% / 62.04% / 71.06% |
| visible surface coverage | 67.91% |
| true-identity conditional spatial-null AUPRC | 24.50% |
| anonymous region-mode Top-1 within 30 cm / 3 deg | 0/12 |
| anonymous region-mode oracle Top-16 within 30 cm / 3 deg | 0/12 |
| Top-1 translation median / P90 | 2.96 m / 8.52 m |
| Top-1 rotation median / P90 | 7.77 deg / 29.16 deg |

This is a failed coarse-pose gate, not a production result. The bearing update
is a small improvement over fixed-pose voting, but it does not solve
cross-trajectory extrapolation.

The oracle decomposition was also corrected. An older evaluator accidentally
read “true-maplet spatial components” from the identity bank, whose component
geometry collapses to the maplet center. It now reads them from the separate
spatial bank:

| Region-center diagnostic | Median translation / rotation | Within 30 cm / 3 deg |
| --- | ---: | ---: |
| nearest exact visible surface | 13.57 cm / 0.33 deg | 10/12 |
| true identity + oracle spatial component | 11.87 cm / 0.48 deg | 11/12 |
| retrieved true identity + oracle spatial component | 27.25 cm / 0.93 deg | 6/12 |
| true identity + descriptor-MAP spatial component | 58.18 cm / 2.01 deg | 2/12 |
| true maplet center | 1.25 m / 3.07 deg | 0/12 |

This corrected split is decisive: clean geometry is not the dominant failure.
Identity/component selection and the region-level pose proposal lose most of
the basin coverage. Even the two oracle rows remain region-center PnP
diagnostics and do not establish the final 4 cm observation model.

Ground-truth pose is also scored strictly after hypothesis generation as a
diagnostic only. With the oriented quad footprint, GT has higher likelihood
than every generated proposal on 10/12 queries. This shows that proposal
coverage/search is the dominant failure on most queries. On 2/12, a
repetitive wrong structure still has higher footprint likelihood, so merely
widening CEM cannot be promoted safely.

A fixed two-query CEM check confirmed that warning: it remained 0/2 inside
30 cm / 3 degrees (Top-1 median 4.29 m / 14.06 degrees), and its optimized
false modes outscored GT footprint likelihood on both queries. The footprint
optimizer is therefore retained as an experimental diagnostic, not selected
as the default coarse method.

The exact-surface and surface-component PnP rows in the dashboard are
observation-model diagnostics. They use RADIO region centers and therefore do
not establish final metric accuracy. Earlier tables that described regional
MAP-PnP as deployable are historical and invalid under the corrected method
definition.

## Promotion policy

The 530-query run is not permitted while the following remain false:

- G0: sufficient adaptive atlas coverage, chart quality and full-scene
  visibility;
- G1: trajectory-disjoint typed-null correlation and reliable flow direction;
- G2: two or three verified fine iterations improve fixed 5/20/30 cm starts;
- G3: actual retrieval and region-level proposal put enough strict queries
  inside the 20–30 cm / 3 degree basin;
- G4: only then run and report all 530 end-to-end queries.

Scientific diagnostics continue below a gate, but a failed component cannot be
renamed production or used to claim final accuracy.

## Active implementation

- `feature_extract/tools/vfm/build_surface_retrieval_maplet_bank.py`
- `feature_extract/tools/vfm/calibrate_v6_retrieval_probabilities.py`
- `feature_extract/tools/vfm/build_v6_anonymous_pose_vote_bank.py`
- `feature_extract/tools/vfm/evaluate_v6_retrieval_pose_basin.py`
- `feature_extract/vfm/localization_v6/maplet_retrieval.py`
- `feature_extract/vfm/localization_v6/maplet_pose_voting.py`
- `feature_extract/vfm/localization_v6/maplet_footprint_pose.py`
- `feature_extract/vfm/localization_v6/maplet_pose_proposal.py`
- `feature_extract/vfm/localization_v6/probability_calibration.py`
- `feature_extract/vfm/localization_v6/atlas_renderer.py`
- `feature_extract/vfm/localization_v6/local_correlation.py`
- `feature_extract/vfm/localization_v6/se3_update.py`
